pipeline {

    agent { label 'chatbot_mlops_214' }

    environment {
        PYTHON      = "/home/thinh/Chatbot_tien/.venv/bin/python"
        VENV_BIN    = "/home/thinh/Chatbot_tien/.venv/bin"

        PROJECT_DIR = "${WORKSPACE}"
        MODEL_DIR   = "${WORKSPACE}/rasa_bot/models"

        LD_LIB      = "/home/thinh/miniconda3/envs/rasa/lib"

        MLFLOW_URI  = "http://127.0.0.1:5000"
        ENV_FILE    = "/home/thinh/Chatbot_tien/.env"
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 90, unit: 'MINUTES')
    }

    stages {

        stage('1. Verify Environment') {
            steps {
                echo "✅ Kiểm tra môi trường Jenkins Agent, Python, Rasa, MLflow..."

                sh """
                    set -e

                    export PATH="${VENV_BIN}:\\$PATH"
                    export LD_LIBRARY_PATH="${LD_LIB}:\\$LD_LIBRARY_PATH"

                    echo "Workspace: ${PROJECT_DIR}"

                    echo "Python version:"
                    ${PYTHON} --version

                    echo "Kiểm tra package chính:"
                    ${PYTHON} -c "import rasa; print('Rasa:', rasa.__version__)"
                    ${PYTHON} -c "import mlflow; print('MLflow:', mlflow.__version__)"

                    echo "Kiểm tra Rasa CLI:"
                    which rasa
                    rasa --version

                    echo "Copy .env vào workspace..."
                    cp "${ENV_FILE}" "${PROJECT_DIR}/.env" || true

                    echo "Kiểm tra MLflow server..."
                    curl -fsS "${MLFLOW_URI}" >/dev/null
                    echo "✅ MLflow server đang chạy tại ${MLFLOW_URI}"
                """
            }
        }

        stage('2. Data Pipeline') {
            steps {
                echo "🔄 Convert CSV sang Rasa format..."

                sh """
                    set -e

                    cd "${PROJECT_DIR}"

                    export PATH="${VENV_BIN}:\\$PATH"
                    export LD_LIBRARY_PATH="${LD_LIB}:\\$LD_LIBRARY_PATH"

                    ${PYTHON} data/csv_to_rasa.py

                    echo "Đồng bộ file NLU generated sang rasa_bot/data nếu có..."
                    if [ -f "data/nlu_test.yml" ]; then
                        cp data/nlu_test.yml rasa_bot/data/nlu_test.yml
                        echo "✅ Đã copy data/nlu_test.yml -> rasa_bot/data/nlu_test.yml"
                    else
                        echo "⚠️ Không tìm thấy data/nlu_test.yml, bỏ qua bước copy"
                    fi

                    echo "✅ Data pipeline hoàn tất"
                """
            }
        }

        stage('3. Train Model') {
            steps {
                echo "🚀 Train Rasa + log MLflow..."

                sh """
                    set -e

                    cd "${PROJECT_DIR}"

                    export PATH="${VENV_BIN}:\\$PATH"
                    export LD_LIBRARY_PATH="${LD_LIB}:\\$LD_LIBRARY_PATH"
                    export MLFLOW_TRACKING_URI="${MLFLOW_URI}"

                    echo "Kiểm tra lại Rasa CLI trước khi train:"
                    which rasa
                    rasa --version

                    ${PYTHON} scripts/train_mlflow.py

                    echo "✅ Train model hoàn tất"
                """
            }
        }

        stage('4. Check Model Artifact') {
            steps {
                echo "🔎 Kiểm tra model artifact..."

                sh """
                    set -e

                    echo "Model directory: ${MODEL_DIR}"
                    ls -lah "${MODEL_DIR}" || true

                    LATEST_MODEL=\\$(ls -t "${MODEL_DIR}"/*.tar.gz 2>/dev/null | head -n 1 || true)

                    if [ -z "\\$LATEST_MODEL" ]; then
                        echo "❌ Không tìm thấy model .tar.gz trong ${MODEL_DIR}"
                        exit 1
                    fi

                    echo "✅ Model mới nhất: \\$LATEST_MODEL"
                    echo "\\$LATEST_MODEL" > latest_model_path.txt
                """
            }
        }

        stage('5. Human Approval') {
            steps {
                script {
                    def latestModel = sh(
                        script: 'cat latest_model_path.txt',
                        returnStdout: true
                    ).trim()

                    echo "🔔 Model đã train xong: ${latestModel}"
                    echo "📊 Vào MLflow kiểm tra metrics trước khi quyết định deploy."

                    def decision = input(
                        id: 'DeployGate',
                        message: "Model: ${latestModel}\nKiểm tra MLflow xong. Deploy không?",
                        ok: 'Submit',
                        parameters: [
                            choice(
                                name: 'DECISION',
                                choices: ['deploy', 'reject'],
                                description: 'deploy = triển khai, reject = dừng pipeline'
                            )
                        ]
                    )

                    if (decision == 'reject') {
                        currentBuild.result = 'ABORTED'
                        error("🛑 Model bị reject. Pipeline dừng, không deploy.")
                    }

                    echo "✅ Model được duyệt. Tiếp tục deploy."
                }
            }
        }

        stage('6. Deploy Model') {
            steps {
                echo "☁️ Deploy model..."

                sh """
                    set -e

                    cd "${PROJECT_DIR}"

                    export PATH="${VENV_BIN}:\\$PATH"
                    export LD_LIBRARY_PATH="${LD_LIB}:\\$LD_LIBRARY_PATH"

                    ${PYTHON} scripts/deploy_model.py

                    echo "✅ Deploy model hoàn tất"
                """
            }
        }
    }

    post {

        always {
            echo "📦 Archive artifacts nếu có..."

            archiveArtifacts artifacts: '''
                latest_model_path.txt,
                error_log.txt,
                rasa_bot/results/**/*,
                rasa_bot/models/*.tar.gz
            ''', allowEmptyArchive: true
        }

        success {
            echo "🎉 PIPELINE HOÀN TẤT!"
        }

        aborted {
            echo "⚠️ Model bị reject hoặc pipeline bị hủy. Không deploy."
        }

        failure {
            echo "🔥 Pipeline thất bại. Kiểm tra Console Output và artifact error_log.txt nếu có."
        }
    }
}