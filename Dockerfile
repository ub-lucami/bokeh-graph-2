FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY first_screen_fileinput.py .

EXPOSE 5006

CMD ["bokeh", "serve", "first_screen_fileinput.py", "--allow-websocket-origin=*", "--port=5006", "--websocket-max-message-size=1073741824"]
