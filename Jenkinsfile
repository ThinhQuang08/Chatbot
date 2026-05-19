pipeline {

    agent { label 'chatbot-mlops' }

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        PROJECT_DIR = "${WORKSPACE}"

        IMAGE_NAME = "chatbot-mlops-runner"
        IMAGE_TAG = "${BUILD_NUMBER}"
        IMAGE_FULL = "chatbot-mlops-runner:${BUILD_NUMBER}"
        IMAGE_LATEST = "chatbot-mlops-runner:latest"

        MODEL_DIR = "${WORKSPACE}/rasa_bot/models"

        MODEL_REGISTRY_DIR = "/opt/chatbot/model-registry"
        ACTIVE_MODEL_POINTER = "/opt/chatbot/model-registry/active_model.txt"
    }

    stages {

        stage('1. Check Agent Environment') {
            steps {
                echo "🔍 Kiểm tra môi trường Jenkins agent và Docker..."

                sh """
                    set -e

                    echo "Current user:"
                    whoami

                    echo "Hostname:"
                    hostname

                    echo "Current workspace:"
                    pwd

                    echo "Git version:"
                    git --version

                    echo "Docker version:"
                    docker --version

                    echo "Docker info:"
                    docker info | head -40

                    echo "Disk usage:"
                    df -h

                    echo "Memory usage:"
                    free -h

                    echo "Workspace content:"
                    ls -la
                """
            }
        }

        stage('2. Build Docker Runner Image') {
            steps {
                echo "🐳 Đang build Docker image cho chatbot MLOps runner..."

                sh """
                    set -e

                    cd "${PROJECT_DIR}"

                    echo "Build image: ${IMAGE_FULL}"
                    docker build -t "${IMAGE_FULL}" -t "${IMAGE_LATEST}" .

                    echo "Test Python inside container:"
                    docker run --rm "${IMAGE_FULL}" python --version

                    echo "Test important packages:"
                    docker run --rm "${IMAGE_FULL}" python - <<'PY'
import sys
print("Python:", sys.version)
import rasa
print("Rasa:", rasa.__version__)
PY
                """
            }
        }

        stage('3. Generate Massive Data') {
            steps {
                echo "📦 Đang tạo / thu thập dữ liệu chat trong Docker..."

                sh """
                    set -e

                    cd "${PROJECT_DIR}"

                    docker run --rm \
                        -v "${PROJECT_DIR}:/app" \
                        -w /app \
                        "${IMAGE_FULL}" \
                        python data/generate_massive_data.py
                """
            }
        }

        stage('4. Preprocess Data') {
            steps {
                echo "🧹 Đang làm sạch và chuẩn hóa dữ liệu trong Docker..."

                sh """
                    set -e

                    cd "${PROJECT_DIR}"

                    docker run --rm \
                        -v "${PROJECT_DIR}:/app" \
                        -w /app \
                        "${IMAGE_FULL}" \
                        python data/preprocess_data.py
                """
            }
        }

        stage('5. Auto Label with Snorkel') {
            steps {
                echo "🏷️ Đang gán nhãn tự động bằng Snorkel trong Docker..."

                sh """
                    set -e

                    cd "${PROJECT_DIR}"

                    docker run --rm \
                        -v "${PROJECT_DIR}:/app" \
                        -w /app \
                        "${IMAGE_FULL}" \
                        python data/auto_label_snorkel.py
                """
            }
        }

        stage('6. Split Confidence') {
            steps {
                echo "📊 Đang tách dữ liệu theo độ tự tin trong Docker..."

                sh """
                    set -e

                    cd "${PROJECT_DIR}"

                    docker run --rm \
                        -v "${PROJECT_DIR}:/app" \
                        -w /app \
                        "${IMAGE_FULL}" \
                        python data/split_confidence.py
                """
            }
        }

        stage('7. Validate with Cleanlab') {
            steps {
                echo "✅ Đang validate nhãn bằng Cleanlab trong Docker..."

                sh """
                    set -e

                    cd "${PROJECT_DIR}"

                    docker run --rm \
                        -v "${PROJECT_DIR}:/app" \
                        -w /app \
                        "${IMAGE_FULL}" \
                        python data/validate_cleanlab.py
                """
            }
        }

        stage('8. Convert CSV to Rasa') {
            steps {
                echo "🔄 Đang chuyển dữ liệu sang định dạng Rasa trong Docker..."

                sh """
                    set -e

                    cd "${PROJECT_DIR}"

                    docker run --rm \
                        -v "${PROJECT_DIR}:/app" \
                        -w /app \
                        "${IMAGE_FULL}" \
                        python data/csv_to_rasa.py
                """
            }
        }

        stage('9. Train Model') {
            steps {
                echo "🚀 Đang huấn luyện Rasa và lưu metrics lên MLflow trong Docker..."

                sh """
                    set -e

                    cd "${PROJECT_DIR}"

                    docker run --rm \
                        -v "${PROJECT_DIR}:/app" \
                        -w /app \
                        "${IMAGE_FULL}" \
                        python scripts/train_mlflow.py
                """
            }
        }

        stage('10. Check Generated Model') {
            steps {
                echo "🔎 Đang kiểm tra model artifact..."

                sh """
                    set -e

                    cd "${PROJECT_DIR}"

                    echo "Listing model directory:"
                    ls -lah "${MODEL_DIR}" || true

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
                    set -e

                    cd "${PROJECT_DIR}"

                    LATEST_MODEL=\$(cat latest_model_path.txt)
                    MODEL_NAME=\$(basename "\$LATEST_MODEL")
                    VERSION_DIR="${MODEL_REGISTRY_DIR}/${BUILD_NUMBER}"

                    sudo mkdir -p "\$VERSION_DIR"
                    sudo cp "\$LATEST_MODEL" "\$VERSION_DIR/\$MODEL_NAME"
                    echo "\$VERSION_DIR/\$MODEL_NAME" | sudo tee "${ACTIVE_MODEL_POINTER}"

                    echo "✅ Active model hiện tại:"
                    cat "${ACTIVE_MODEL_POINTER}"
                """
            }
        }

        stage('13. Update Rasa Endpoint') {
            steps {
                echo "☁️ Đang cập nhật Rasa sang model mới trong Docker..."

                sh """
                    set -e

                    cd "${PROJECT_DIR}"

                    docker run --rm \
                        -v "${PROJECT_DIR}:/app" \
                        -v "${MODEL_REGISTRY_DIR}:${MODEL_REGISTRY_DIR}" \
                        -w /app \
                        "${IMAGE_FULL}" \
                        python scripts/deploy_model.py
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

            echo "🧹 Dọn container rác nếu có..."
            sh """
                docker container prune -f || true
            """

            echo "🏁 Kết thúc pipeline."
        }
    }
}