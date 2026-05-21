pipeline {

    agent { label 'chatbot-mlops' }

    environment {
        PYTHON        = "/home/thinh/Chatbot_tien/.venv/bin/python"
        WORKSPACE_DIR = "${WORKSPACE}"
        MODEL_DIR     = "${WORKSPACE}/rasa_bot/models"
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 60, unit: 'MINUTES')
    }

    stages {

        stage('1. Verify Environment') {
            steps {
                echo "✅ Kiểm tra môi trường có sẵn..."
                sh """
                    set -e
                    ${PYTHON} --version
                    ${PYTHON} -c "import rasa; print('Rasa:', rasa.__version__)"
                    ${PYTHON} -c "import mlflow; print('MLflow:', mlflow.__version__)"
                    ${PYTHON} -c "import sentence_transformers; print('SentenceTransformers:', sentence_transformers.__version__)"
                """
            }
        }

        stage('2. Data Pipeline') {
            steps {
                echo "🧹 Chạy data pipeline..."
                sh """
                    set -e
                    cd "${WORKSPACE_DIR}"
                    ${PYTHON} data/generate_massive_data.py
                    ${PYTHON} data/preprocess_data.py
                    ${PYTHON} data/auto_label_snorkel.py
                    ${PYTHON} data/split_confidence.py
                    ${PYTHON} data/validate_cleanlab.py
                    ${PYTHON} data/csv_to_rasa.py
                """
            }
        }

        stage('3. Train Model') {
            steps {
                echo "🚀 Train Rasa + log MLflow..."
                sh """
                    set -e
                    cd "${WORKSPACE_DIR}"
                    ${PYTHON} scripts/train_mlflow.py
                """
            }
        }

        stage('4. Check Model Artifact') {
            steps {
                echo "🔎 Kiểm tra model artifact..."
                sh """
                    set -e

                    echo "Model directory:"
                    ls -lah "${MODEL_DIR}" || true

                    LATEST_MODEL=\$(ls -t "${MODEL_DIR}"/*.tar.gz 2>/dev/null | head -n 1 || true)

                    if [ -z "\$LATEST_MODEL" ]; then
                        echo "❌ Không tìm thấy model .tar.gz"
                        exit 1
                    fi

                    echo "✅ Model: \$LATEST_MODEL"
                    echo "\$LATEST_MODEL" > latest_model_path.txt
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
                    echo "📊 Vào MLflow kiểm tra metrics trước khi quyết định."

                    def decision = input(
                        id: 'DeployGate',
                        message: "Deploy model: ${latestModel}?",
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
                        error("🛑 Model bị reject. Pipeline dừng.")
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
                    cd "${WORKSPACE_DIR}"
                    ${PYTHON} scripts/deploy_model.py
                """
            }
        }
    }

    post {
        success {
            echo "🎉 PIPELINE HOÀN TẤT!"
            archiveArtifacts artifacts: 'latest_model_path.txt', allowEmptyArchive: true
        }
        aborted {
            echo "⚠️ Model bị reject. Không deploy."
        }
        failure {
            echo "🔥 Pipeline thất bại. Kiểm tra console log."
        }
    }
}