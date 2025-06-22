FROM tensorflow/serving:latest

COPY ./serving_model /models
COPY ./monitoring/prometheus.config /model_config/prometheus.config

# Define environment variables
ENV MODEL_NAME=cinnamon-classification-model
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG=/model_config/prometheus.config
ENV PORT=8501

# Create entrypoint script
RUN echo '#!/bin/bash\n\n\
env\n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \\\n\
--model_name=${MODEL_NAME} \\\n\
--model_base_path=${MODEL_BASE_PATH} \\\n\
--monitoring_config_file=${MONITORING_CONFIG} \\\n\
"$@"' > /usr/bin/tf_serving_entrypoint.sh && \
chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
