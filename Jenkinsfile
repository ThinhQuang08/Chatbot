pipeline {
    agent any

    environment {
        // Venv nằm ngoài workspace → tồn tại giữa các lần build, không bị xóa
        VENV_DIR   = "/var/lib/jenkins/.rasa-venv"
        PY         = "/var/lib/jenkins/.rasa-venv/bin/python"
        PIP        = "/var/lib/jenkins/.rasa-venv/bin/pip"

        // Docker image trên DockerHub
        DOCKER_IMAGE = "mnhat1/chatbot-rasa"

        // k8s-manifests GitOps repo
        K8S_MANIFESTS_DIR = "/var/lib/jenkins/k8s-manifests"

        // AWS S3 — model artifact
        CHATBOT_S3_BUCKET    = "kltn-chatbot-artifacts-dev"
        CHATBOT_S3_MODEL_KEY = "models/latest_model.tar.gz"
        AWS_DEFAULT_REGION   = "ap-southeast-1"

        // MLflow (chạy cùng EC2 Jenkins hoặc localhost)
        MLFLOW_TRACKING_URI = "http://localhost:5000"
        MLFLOW_EXPERIMENT   = "Travel_Chatbot_Rasa"
    }

    stages {

        // ─────────────────────────────────────────────────
        // STAGE 0: Cài đặt môi trường Python (cache lại giữa các build)
        // ─────────────────────────────────────────────────
        stage('0. Setup Python Env') {
            steps {
                echo "📦 Cài python3.10-venv (Rasa 3.6 yêu cầu Python 3.8-3.10, Ubuntu 24.04 là 3.12)..."
                sh """
                    set -e
                    
                    # Kiểm tra xem venv có hợp lệ và đã cài rasa chưa
                    if ! "${PY}" -c "import rasa" > /dev/null 2>&1; then
                        echo "🔧 Rasa chưa được cài đặt hoặc venv hỏng. Tái tạo virtualenv tại ${VENV_DIR} với Python 3.10..."
                        rm -rf "${VENV_DIR}"
                        python3.10 -m venv "${VENV_DIR}"
                        
                        echo "📦 Đang tải và cài đặt dependencies (có thể tốn vài phút)..."
                        "${PIP}" install --upgrade pip --quiet
                        "${PIP}" install dvc dvc-s3 pandas boto3 python-dotenv mlflow pyyaml --quiet
                        "${PIP}" install -r requirements.txt --quiet
                    else
                        echo "✅ Môi trường Python 3.10 và Rasa đã sẵn sàng. Bỏ qua bước cài đặt."
                    fi

                    echo "📦 Đảm bảo DVC và AWS CLI luôn được cài đặt..."
                    "${PIP}" install dvc dvc-s3 awscli --quiet

                    echo "✅ Python env sẵn sàng: \$(${PY} --version)"
                """
            }
        }

        // ─────────────────────────────────────────────────
        // STAGE 0.1: Code & Library Validation
        // ─────────────────────────────────────────────────
        stage('0.1. Validation') {
            steps {
                echo "🔍 Đang kiểm tra mã nguồn và thư viện..."
                sh """
                    set -e
                    export PATH="${VENV_DIR}/bin:\$PATH"
                    
                    echo "1️⃣ Validate Security (Trivy Vulnerability Scanner)..."
                    "${PIP}" install flake8 --quiet
                    
                    if [ ! -f "${VENV_DIR}/bin/trivy" ]; then
                        echo "⬇️ Đang tải Trivy qua official script..."
                        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b "${VENV_DIR}/bin"
                    fi
                    
                    # Quét lỗ hổng thư viện và file cấu hình (Trivy)
                    # exit-code 0 để không làm sập pipeline nếu phát hiện lỗ hổng cũ của Rasa
                    trivy fs . --scanners vuln,secret --severity HIGH,CRITICAL --exit-code 0
                    
                    echo "2️⃣ Validate Code (Linting Python scripts)..."
                    # Flake8: Phát hiện lỗi cú pháp nghiêm trọng (sẽ block pipeline nếu có lỗi Syntax)
                    flake8 scripts/ data/ database/ --count --select=E9,F63,F7,F82 --show-source --statistics
                    
                    # Flake8: Phát hiện cảnh báo style code (không block pipeline)
                    flake8 scripts/ data/ database/ --count --exit-zero --max-complexity=15 --max-line-length=127 --statistics
                    
                    echo "✅ Validation hoàn tất mượt mà!"
                """
            }
        }

        // ─────────────────────────────────────────────────
        // STAGE 1: Data Pipeline — DVC Pull & Snorkel label
        // ─────────────────────────────────────────────────
        stage('1. Data Pipeline') {
            steps {
                echo "🧹 DVC Pull dữ liệu từ S3 và chạy csv_to_rasa.py..."
                withCredentials([
                    string(credentialsId: 'aws-access-key-id',     variable: 'AWS_ACCESS_KEY_ID'),
                    string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
                ]) {
                    sh """
                        set -e
                        export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
                        export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
                        export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}"
                        export PYTHONPATH="."
                        export PATH="${VENV_DIR}/bin:\$PATH"
                        
                        echo "⬇️ Kéo dữ liệu từ AWS S3 thông qua DVC..."
                        "${VENV_DIR}/bin/dvc" pull
                        
                        echo "🔄 Chạy tiền xử lý dữ liệu..."
                        "${PY}" data/csv_to_rasa.py
                    """
                }
            }
        }

        // ─────────────────────────────────────────────────
        // STAGE 2: Train Model + Log MLflow + Upload S3
        // train_mlflow.py sẽ: rasa train → evaluate → log MLflow → upload S3
        // ─────────────────────────────────────────────────
        stage('2. Train Model') {
            steps {
                echo "🚀 Train Rasa model và log metrics lên MLflow..."
                withCredentials([
                    string(credentialsId: 'aws-access-key-id',     variable: 'AWS_ACCESS_KEY_ID'),
                    string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
                ]) {
                    sh """
                        set -e
                        export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
                        export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
                        export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}"
                        export MLFLOW_TRACKING_URI="file://\$(pwd)/mlruns"
                        export MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT}"
                        export CHATBOT_S3_BUCKET="${CHATBOT_S3_BUCKET}"
                        export CHATBOT_S3_MODEL_KEY="${CHATBOT_S3_MODEL_KEY}"
                        export PYTHONPATH="."
                        export PATH="${VENV_DIR}/bin:\$PATH"

                        "${PY}" scripts/train_mlflow.py
                    """
                }
            }
        }

        // ─────────────────────────────────────────────────
        // STAGE 3: Human Approval — review MLflow trước khi deploy
        // ─────────────────────────────────────────────────
        stage('3. Human Approval') {
            steps {
                script {
                    echo "🔔 Xem metrics tại ${MLFLOW_TRACKING_URI} rồi quyết định..."
                    def decision = input(
                        id: 'DeployGate',
                        message: '📊 Metrics đã có trên MLflow. Quyết định deploy?',
                        ok: 'Deploy',
                        parameters: [
                            choice(
                                name: 'DECISION',
                                choices: ['deploy', 'reject'],
                                description: 'deploy = tiếp tục build Docker image, reject = dừng pipeline'
                            )
                        ]
                    )
                    if (decision == 'reject') {
                        error("🛑 Model bị reject bởi người dùng — pipeline dừng lại.")
                    }
                    echo "✅ Model được duyệt — bắt đầu build Docker image"
                }
            }
        }

        // ─────────────────────────────────────────────────
        // STAGE 4: Build & Push Docker Image
        // Download model từ S3 → bake vào image → push DockerHub
        // ─────────────────────────────────────────────────
        stage('4. Build & Push Docker Image') {
            steps {
                script {
                    def gitSha = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    env.IMAGE_TAG  = "dev-${BUILD_NUMBER}-${gitSha}"
                    env.FULL_IMAGE = "${DOCKER_IMAGE}:${env.IMAGE_TAG}"
                    echo "🐳 Building image: ${env.FULL_IMAGE}"

                    // Download model từ S3 → thư mục rasa_bot/models/
                    withCredentials([
                        string(credentialsId: 'aws-access-key-id',     variable: 'AWS_ACCESS_KEY_ID'),
                        string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
                    ]) {
                        sh """
                            set -e
                            export PATH="${VENV_DIR}/bin:\$PATH"
                            mkdir -p rasa_bot/models
                            echo "⬇️  Downloading model từ S3..."
                            aws s3 cp s3://${CHATBOT_S3_BUCKET}/${CHATBOT_S3_MODEL_KEY} \\
                                rasa_bot/models/latest_model.tar.gz \\
                                --region ${AWS_DEFAULT_REGION}
                            echo "✅ Download xong: \$(du -sh rasa_bot/models/latest_model.tar.gz)"
                        """
                    }

                    // Build Docker image (model đã có sẵn trong workspace)
                    withCredentials([usernamePassword(
                        credentialsId: 'dockerhub-credentials',
                        usernameVariable: 'DOCKER_USER',
                        passwordVariable: 'DOCKER_PASS'
                    )]) {
                        sh """
                            set -e
                            echo "\$DOCKER_PASS" | docker login -u "\$DOCKER_USER" --password-stdin
                            docker build -f Dockerfile.prod -t ${env.FULL_IMAGE} .
                            docker push ${env.FULL_IMAGE}
                            docker rmi ${env.FULL_IMAGE} || true
                        """
                    }
                    echo "✅ Image đã push lên DockerHub: ${env.FULL_IMAGE}"
                }
            }
        }

        // ─────────────────────────────────────────────────
        // STAGE 5: Update k8s-manifests → GitOps → ArgoCD sync
        // ─────────────────────────────────────────────────
        stage('5. Update k8s-manifests') {
            steps {
                script {
                    echo "📝 Cập nhật image tag trong k8s-manifests repo..."
                    withCredentials([usernamePassword(
                        credentialsId: 'github',
                        usernameVariable: 'GH_USER',
                        passwordVariable: 'GH_TOKEN'
                    )]) {
                        sh """
                            set -e

                            # Sử dụng GIT_ASKPASS để xử lý mật khẩu chứa ký tự đặc biệt (@, #, !)
                            cat << 'EOF' > "\${WORKSPACE}/git-askpass.sh"
#!/bin/sh
echo "\$GH_TOKEN"
EOF
                            chmod +x "\${WORKSPACE}/git-askpass.sh"
                            export GIT_ASKPASS="\${WORKSPACE}/git-askpass.sh"
                            export GIT_USERNAME="\${GH_USER}"

                            # Clone nếu chưa có, pull nếu đã có
                            if [ -d "${K8S_MANIFESTS_DIR}/.git" ]; then
                                cd "${K8S_MANIFESTS_DIR}"
                                git pull origin main
                            else
                                rm -rf "${K8S_MANIFESTS_DIR}"
                                git clone https://github.com/minhnhatuit734/k8s-manifests.git "${K8S_MANIFESTS_DIR}"
                                cd "${K8S_MANIFESTS_DIR}"
                            fi
                            
                            # Xóa script sau khi dùng xong
                            rm -f "\${WORKSPACE}/git-askpass.sh"

                            cd "${K8S_MANIFESTS_DIR}"
                            git config user.email "jenkins@kltn.local"
                            git config user.name "Jenkins Bot"

                            # Dùng kustomize để update image tag
                            cd overlays/dev
                            kustomize edit set image mnhat1/chatbot-rasa=${env.FULL_IMAGE}
                            cd "${K8S_MANIFESTS_DIR}"

                            git add overlays/dev/kustomization.yaml
                            git diff --cached --quiet && echo "No changes to commit" || \\
                                git commit -m "chatbot: update rasa image to ${env.IMAGE_TAG} [skip ci]"

                            git push origin main
                        """
                    }
                    echo "🚀 ArgoCD sẽ tự động sync image mới vào EKS!"
                }
            }
        }
    }

    post {
        success {
            echo "🎉 PIPELINE HOÀN TẤT! Image: ${env.FULL_IMAGE ?: 'N/A'}"
        }
        aborted {
            echo "⚠️ Pipeline bị hủy bởi người dùng"
            sh "rm -f rasa_bot/models/latest_model.tar.gz || true"
        }
        failure {
            echo "🔥 Pipeline thất bại — xem log bên trên để debug"
            sh "rm -f rasa_bot/models/latest_model.tar.gz || true"
        }
        cleanup {
            // Luôn đăng xuất Docker sau mỗi lần build
            sh "docker logout || true"
        }
    }
}