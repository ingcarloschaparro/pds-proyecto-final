FROM pls-api-base AS runner

WORKDIR /app

COPY README.md ./
COPY src/ src/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

RUN groupadd -r noroot && useradd -r -m -g noroot noroot
RUN chown -R noroot:noroot /app

USER noroot

EXPOSE 9000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "9000"]

##docker build -t pls-api:latest .
##docker run --name pls-api -p 8001:9000 -d -it pls-api:latest

##docker stop pls-api && docker rm pls-api
##docker rmi pls-api:latest
