pipeline {

    agent { label 'chatbot-mlops' }

    environment {
        WORKSPACE_DIR = "${WORKSPACE}"
        VENV_DIR      = "${WORKSPACE}/.venv"
        PYTHON        = "${WORKSPACE}/.venv/bin/python"
        MODEL_DIR     = "${WORKSPACE}/rasa_bot/models"
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 60, unit: 'MINUTES')
    }

    stages {

        stage('1. Setup Environment') {
            steps {
                echo "🔧 Chuẩn bị môi trường Python..."
                sh """
                    set -e
                    cd "${WORKSPACE_DIR}"

                    # Dùng Python 3.10 từ pyenv đã cài
                    export PATH="/home/jenkins/.pyenv/versions/3.10.14/bin:\$PATH"

                    # Tạo .venv nếu chưa có
                    if [ ! -f "${VENV_DIR}/bin/python" ]; then
                        echo "Tạo .venv mới..."
                        python -m venv "${VENV_DIR}"
                    else
                        echo ".venv đã tồn tại, bỏ qua tạo mới."
                    fi

                    # Cài/update dependencies
                    "${VENV_DIR}/bin/pip" install --upgrade pip --quiet
                    "${VENV_DIR}/bin/pip" install --no-cache-dir -r requirements.txt --quiet

                    echo "Python version:"
                    "${PYTHON}" --version

                    echo "Rasa version:"
                    "${PYTHON}" -c "import rasa; print(rasa.__version__)"
                """
            }
        }

        stage('2. Data Pipeline') {
            steps {
                echo "🧹 Generate → Preprocess → Snorkel → Cleanlab → Convert CSV..."
                sh """
                    set -e
                    cd "${WORKSPACE_DIR}"

                    "${PYTHON}" data/generate_massive_data.py
                    "${PYTHON}" data/preprocess_data.py
                    "${PYTHON}" data/auto_label_snorkel.py
                    "${PYTHON}" data/split_confidence.py
                    "${PYTHON}" data/validate_cleanlab.py
                    "${PYTHON}" data/csv_to_rasa.py
                """
            }
        }

        stage('3. Train Model') {
            steps {
                echo "🚀 Train Rasa + log metrics lên MLflow..."
                sh """
                    set -e
                    cd "${WORKSPACE_DIR}"

                    "${PYTHON}" scripts/train_mlflow.py

                    echo "Model directory sau train:"
                    ls -lah "${MODEL_DIR}" || true
                """
            }
        }

        stage('4. Check Model Artifact') {
            steps {
                echo "🔎 Kiểm tra model artifact..."
                sh """
                    set -e

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
                    echo "📊 Vào MLflow kiểm tra metrics trước khi duyệt."

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

                    "${PYTHON}" scripts/deploy_model.py
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
            echo "⚠️ Pipeline bị hủy hoặc model bị reject."
        }
        failure {
            echo "🔥 Pipeline thất bại. Kiểm tra console log."
        }
        always {
            echo "🧹 Cleanup..."
            sh "docker container prune -f || true"
        }
    }
}