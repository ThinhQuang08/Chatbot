pipeline {
    agent any

    environment {
        // Venv nằm ngoài workspace -> tồn tại giữa các lần build, không bị xóa
        VENV_DIR   = "/var/lib/jenkins/.rasa-venv"
        PY         = "/var/lib/jenkins/.rasa-venv/bin/python"
        PIP        = "/var/lib/jenkins/.rasa-venv/bin/pip"

        // workspace thật của Jenkins
        PROJECT_DIR = "${WORKSPACE}"
        MODEL_DIR   = "${WORKSPACE}/rasa_bot/models"

        // Private IP của Chatbot EC2 (vì hệ thống đang chạy trên 2 EC2 riêng biệt)
        CHATBOT_HOST = "10.0.1.5"

        MLFLOW_TRACKING_URI = "http://${CHATBOT_HOST}:5000"
        MLFLOW_S3_ENDPOINT_URL = "http://${CHATBOT_HOST}:9000"
        MINIO_URL = "http://${CHATBOT_HOST}:9000"
        MINIO_ENDPOINT = "http://${CHATBOT_HOST}:9000"
        RASA_API_URL = "http://${CHATBOT_HOST}:5005"

        // AWS/MinIO credentials for boto3 and mlflow artifact upload
        AWS_ACCESS_KEY_ID = "admin"
        AWS_SECRET_ACCESS_KEY = "password123"
        AWS_DEFAULT_REGION = "ap-southeast-1"
    }

    stages {
        // ─────────────────────────────────────────────────
        // STAGE 0: Setup Python Environment
        // ─────────────────────────────────────────────────
        stage('0. Setup Python Env') {
            steps {
                echo "📦 Cài đặt môi trường ảo (virtualenv) và các dependencies..."
                sh """
                    set -e
                    cd "${WORKSPACE}"
                    
                    if ! "${PY}" -c "import rasa" > /dev/null 2>&1; then
                        echo "🔧 Rasa chưa được cài đặt hoặc venv bị lỗi. Tạo lại virtualenv tại ${VENV_DIR}..."
                        rm -rf "${VENV_DIR}"
                        python3 -m venv "${VENV_DIR}"
                        
                        echo "📦 Đang cài đặt thư viện..."
                        "${PIP}" install --upgrade pip --quiet
                        # Cài đặt requirements.txt
                        if [ -f "requirements.txt" ]; then
                            "${PIP}" install -r requirements.txt --quiet
                        fi
                        # Bổ sung các thư viện cần thiết cho CI
                        "${PIP}" install flake8 dvc pandas boto3 python-dotenv mlflow pyyaml --quiet
                    else
                        echo "✅ Môi trường Python đã sẵn sàng."
                    fi
                """
            }
        }

        // ─────────────────────────────────────────────────
        // STAGE 0.1: Code & Library Validation
        // ─────────────────────────────────────────────────
        stage('0.1. Validation & Scan') {
            steps {
                echo "🔍 Đang kiểm tra mã nguồn (Linting) và quét lỗ hổng bảo mật..."
                sh """
                    set -e
                    cd "${WORKSPACE}"
                    export PATH="${VENV_DIR}/bin:\$PATH"
                    
                    echo "1️⃣ Quét lỗ hổng bảo mật (Trivy)..."
                    if ! command -v trivy >/dev/null 2>&1; then
                        echo "⬇️ Đang tải Trivy scanner..."
                        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b "${VENV_DIR}/bin"
                    fi
                    
                    # Quét toàn bộ repo (chỉ warning HIGH, CRITICAL, exit-code 0 để không chặn pipeline nếu lỗi nhỏ)
                    trivy fs . --scanners vuln,secret --severity HIGH,CRITICAL --exit-code 0
                    
                    echo "2️⃣ Validate Code (Flake8)..."
                    # Kiểm tra lỗi cú pháp (Syntax errors) - block pipeline nếu có
                    flake8 scripts/ data/ --count --select=E9,F63,F7,F82 --show-source --statistics
                    
                    # Cảnh báo format/style code (không block)
                    flake8 scripts/ data/ --count --exit-zero --max-complexity=15 --max-line-length=127 --statistics
                    
                    echo "✅ Quét hoàn tất!"
                """
            }
        }

        // ─────────────────────────────────────────────────
        // STAGE 1: Data Pipeline
        // ─────────────────────────────────────────────────
        stage('1. Data Pipeline') {
            steps {
                echo "🧹 Đang làm sạch, gán nhãn bằng Snorkel và Validate..."
                sh """
                    cd "${WORKSPACE}"
                    export PATH="${VENV_DIR}/bin:\$PATH"
                    ${PY} data/csv_to_rasa.py
                """
            }
        }

        // ─────────────────────────────────────────────────
        // STAGE 2: Train Model
        // ─────────────────────────────────────────────────
        stage('2. Train Model') {
            steps {
                echo "🚀 Đang huấn luyện Rasa và lưu metrics lên MLflow..."
                sh """
                    cd "${WORKSPACE}"
                    export PATH="${VENV_DIR}/bin:\$PATH"
                    ${PY} scripts/train_mlflow.py
                """
            }
        }

        // ─────────────────────────────────────────────────
        // STAGE 3: Human Approval
        // ─────────────────────────────────────────────────
        stage('3. Human Approval (Gửi TN)') {
            steps {
                script {
                    echo "🔔 Đang chờ sếp kiểm tra thông số trên MLflow..."
                    def userInput = input(
                        id: 'DeployGate',
                        message: 'Thông số mô hình đã có trên MLflow. Sếp quyết định sao?',
                        ok: 'Deploy',
                        parameters: [
                            choice(
                                name: 'DECISION',
                                choices: ['oke_deploy', 'nhu_cc_xoa'],
                                description: 'Chọn hành động'
                            )
                        ]
                    )

                    if (userInput == 'nhu_cc_xoa') {
                        error("🛑 Mô hình bị reject")
                    }

                    echo "✅ Model được duyệt"
                }
            }
        }

        // ─────────────────────────────────────────────────
        // STAGE 4: Deploy to MinIO & Rasa
        // ─────────────────────────────────────────────────
        stage('4. Deploy to MinIO & Rasa') {
            steps {
                echo "☁️ Đang deploy model..."
                sh """
                    cd "${WORKSPACE}"
                    export PATH="${VENV_DIR}/bin:\$PATH"
                    ${PY} scripts/deploy_model.py
                """
            }
        }
    }

    post {
        success {
            echo "🎉 PIPELINE HOÀN TẤT!"
        }
        aborted {
            echo "⚠️ Pipeline bị hủy"
            sh "rm -f \"${WORKSPACE}/rasa_bot/models/*.tar.gz\" || true"
        }
        failure {
            echo "🔥 Pipeline thất bại"
            sh "rm -f \"${WORKSPACE}/rasa_bot/models/*.tar.gz\" || true"
            echo "🗑️ Đã cleanup file rác."
        }
    }
}