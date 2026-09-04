FROM vllm/vllm-openai:latest

COPY entrypoint.sh /opt/ocr/entrypoint.sh
RUN chmod +x /opt/ocr/entrypoint.sh

ENTRYPOINT ["/opt/ocr/entrypoint.sh"]

