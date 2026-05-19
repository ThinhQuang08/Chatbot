pipeline {

    agent { label 'chatbot-mlops' }

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        // Dùng python3 cho chắc, hoặc đổi thành python nếu agent của bạn alias sẵn
        PYTHON_CMD = 'python3'

        // Workspace thật do Jenkins cấp
        PROJECT_DIR = "${WORKSPACE}"

        MODEL_DIR = "${WORKSPACE}/rasa_bot/models"

        // On-prem model registry tạm thời
        MODEL_REGISTRY_DIR = "/opt/chatbot/model-registry"
        ACTIVE_MODEL_POINTER = "/opt/chatbot/model-registry/active_model.txt"
    }

    stages {



        stage('1. Check Agent Environment') {
            steps {
                echo "🔍 Kiểm tra môi trường trên Jenkins agent..."

                sh """
                    echo "Current user:"
                    whoami

                    echo "Hostname:"
                    hostname

                    echo "Current workspace:"
                    pwd

                    echo "Python version:"
                    ${PYTHON_CMD} --version

                    echo "Git version:"
                    git --version

                    echo "Workspace content:"
                    ls -la
                """
            }
        }

        stage('2. Setup Python Environment') {
            steps {
                echo "🐍 Đang chuẩn bị Python virtual environment..."

                sh """
                    cd "${PROJECT_DIR}"

                    ${PYTHON_CMD} -m venv .venv

                    .venv/bin/python -m pip install --upgrade pip

                    if [ -f requirements.txt ]; then
                        .venv/bin/pip install -r requirements.txt
                    else
                        echo "⚠️ Không tìm thấy requirements.txt"
                    fi
                """
            }
        }

        stage('3. Generate Massive Data') {
            steps {
                echo "📦 Đang tạo / thu thập dữ liệu chat..."

                sh """
                    cd "${PROJECT_DIR}"
                    .venv/bin/python data/generate_massive_data.py
                """
            }
        }

        stage('4. Preprocess Data') {
            steps {
                echo "🧹 Đang làm sạch và chuẩn hóa dữ liệu..."

                sh """
                    cd "${PROJECT_DIR}"
                    .venv/bin/python data/preprocess_data.py
                """
            }
        }

        stage('5. Auto Label with Snorkel') {
            steps {
                echo "🏷️ Đang gán nhãn tự động bằng Snorkel..."

                sh """
                    cd "${PROJECT_DIR}"
                    .venv/bin/python data/auto_label_snorkel.py
                """
            }
        }

        stage('6. Split Confidence') {
            steps {
                echo "📊 Đang tách dữ liệu theo độ tự tin..."

                sh """
                    cd "${PROJECT_DIR}"
                    .venv/bin/python data/split_confidence.py
                """
            }
        }

        stage('7. Validate with Cleanlab') {
            steps {
                echo "✅ Đang validate nhãn bằng Cleanlab..."

                sh """
                    cd "${PROJECT_DIR}"
                    .venv/bin/python data/validate_cleanlab.py
                """
            }
        }

        stage('8. Convert CSV to Rasa') {
            steps {
                echo "🔄 Đang chuyển dữ liệu sang định dạng Rasa..."

                sh """
                    cd "${PROJECT_DIR}"
                    .venv/bin/python data/csv_to_rasa.py
                """
            }
        }

        stage('9. Train Model') {
            steps {
                echo "🚀 Đang huấn luyện Rasa và lưu metrics lên MLflow..."

                sh """
                    cd "${PROJECT_DIR}"
                    .venv/bin/python scripts/train_mlflow.py
                """
            }
        }

        stage('10. Check Generated Model') {
            steps {
                echo "🔎 Đang kiểm tra model artifact..."

                sh """
                    cd "${PROJECT_DIR}"

                    LATEST_MODEL=\$(ls -t "${MODEL_DIR}"/*.tar.gz 2>/dev/null | head -n 1 || true)

                    if [ -z "\$LATEST_MODEL" ]; then
                        echo "❌ Không tìm thấy model .tar.gz trong ${MODEL_DIR}"
                        exit 1
                    fi

                    echo "✅ Latest model: \$LATEST_MODEL"
                    echo "\$LATEST_MODEL" > latest_model_path.txt
                """
            }
        }

        stage('11. Human Approval Before Deploy') {
            steps {
                script {

                    def latestModel = sh(
                        script: 'cat latest_model_path.txt',
                        returnStdout: true
                    ).trim()

                    echo "🔔 Model đã train xong."
                    echo "📦 Model artifact: ${latestModel}"
                    echo "📊 Vui lòng kiểm tra metrics/report trên MLflow trước khi deploy."

                    def userInput = input(
                        id: 'DeployGate',
                        message: "Model mới đã sẵn sàng: ${latestModel}. Có deploy model này không?",
                        ok: 'Submit',
                        parameters: [
                            choice(
                                name: 'DECISION',
                                choices: ['deploy', 'reject'],
                                description: 'deploy = triển khai model mới, reject = dừng pipeline'
                            )
                        ]
                    )

                    if (userInput == 'reject') {
                        currentBuild.result = 'ABORTED'
                        error("🛑 Model bị reject. Pipeline dừng, không deploy.")
                    }

                    echo "✅ Model được duyệt. Tiếp tục deploy."
                }
            }
        }

        stage('12. Deploy Model On-Prem') {
            steps {
                echo "📦 Đang lưu model vào model registry on-prem..."

                sh """
                    cd "${PROJECT_DIR}"

                    LATEST_MODEL=\$(cat latest_model_path.txt)
                    MODEL_NAME=\$(basename "\$LATEST_MODEL")
                    VERSION_DIR="${MODEL_REGISTRY_DIR}/${BUILD_NUMBER}"

                    mkdir -p "\$VERSION_DIR"

                    cp "\$LATEST_MODEL" "\$VERSION_DIR/\$MODEL_NAME"

                    echo "\$VERSION_DIR/\$MODEL_NAME" > "${ACTIVE_MODEL_POINTER}"

                    echo "✅ Active model hiện tại:"
                    cat "${ACTIVE_MODEL_POINTER}"
                """
            }
        }

        stage('13. Update Rasa Endpoint') {
            steps {
                echo "☁️ Đang cập nhật Rasa sang model mới..."

                sh """
                    cd "${PROJECT_DIR}"
                    .venv/bin/python scripts/deploy_model.py
                """
            }
        }
    }

    post {

        success {
            echo "🎉 PIPELINE HOÀN TẤT!"
        }

        aborted {
            echo "⚠️ Pipeline bị hủy hoặc model bị reject. Không deploy."
        }

        failure {
            echo "🔥 Pipeline thất bại. Kiểm tra console log."
        }

        always {
            echo "📌 Lưu artifact cần thiết nếu có..."

            archiveArtifacts artifacts: 'latest_model_path.txt', allowEmptyArchive: true

            echo "🏁 Kết thúc pipeline."
        }
    }
}