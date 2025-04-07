import os
import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from collections import Counter
import aiofiles
from fastapi import Request

from tfidf import tokenize, tf, idf, tfidf, visualize_tfidf

app = FastAPI()
templates = Jinja2Templates(directory="templates")

os.makedirs("static", exist_ok=True)


@app.post('/upload', response_class=HTMLResponse)
async def upload(file: UploadFile, request: Request):
    try:
        contents = await file.read()
        async with aiofiles.open(file.filename, 'wb') as f:
            await f.write(contents)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Произошла ошибка при загрузке файла',
        )
    finally:
        list_of_words = tokenize(contents.decode('utf-8'))
        word_count = Counter(list_of_words)
        tfs = tf(word_count)
        idfs = idf(word_count)
        tfidfs = tfidf(word_count, idfs)

        sorted_words = sorted(idfs.items(), key=lambda x: x[1], reverse=True)[:50]

        # DataFrame for visualization
        df = pd.DataFrame(sorted_words, columns=['Word', 'IDF'])
        df['TF'] = df['Word'].apply(lambda x: tfs.get(x, 0))
        df['TF-IDF'] = df['Word'].apply(lambda x: tfidfs.get(x, 0))

        # Heatmap of TF-IDF values
        tfidf_df = pd.DataFrame([tfidfs], index=["Document"], columns=df['Word'])
        heatmap_image_path = "static/tfidf_heatmap.png"
        visualize_tfidf(tfidf_df, heatmap_image_path)

        table_data = df.to_dict(orient='records')

        await file.close()

    return templates.TemplateResponse("table.html", {"request": request, "table_data": table_data, "image_filename": heatmap_image_path})

@app.get('/static/{filename}')
async def get_image(filename: str):
    file_path = os.path.join("static", filename)
    return FileResponse(file_path)

@app.get('/')
async def main():
    content = '''
    <body>
    <form action='/upload' enctype='multipart/form-data' method='post'>
    <input name='file' type='file'>
    <input type='submit'>
    </form>
    </body>
    '''
    return HTMLResponse(content=content)
