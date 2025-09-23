FROM pls-api-base AS runner

WORKDIR /app

COPY README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
COPY start-app.sh start-app.sh

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

RUN groupadd -r noroot && useradd -r -m -g noroot noroot
RUN chown -R noroot:noroot /app
RUN chmod 755 start-app.sh

USER noroot

EXPOSE 9000
EXPOSE 8501

CMD ["./start-app.sh"]

##docker build -t pls-api:latest .
##docker run --name pls-api -p 8001:9000 -p 8501:8501 -d -it pls-api:latest

##docker stop pls-api && docker rm pls-api
##docker rmi pls-api:latest
