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
                    
                    # Tạo venv với Python 3.10
                    if [ ! -f "${VENV_DIR}/bin/activate" ]; then
                        echo "🔧 Tạo virtualenv mới tại ${VENV_DIR} với Python 3.10..."
                        rm -rf "${VENV_DIR}"
                        python3.10 -m venv "${VENV_DIR}"
                    fi

                    # Cài / cập nhật dependencies
                    ${PIP} install --upgrade pip --quiet
                    ${PIP} install pandas boto3 python-dotenv mlflow pyyaml --quiet
                    ${PIP} install -r requirements.txt --quiet

                    echo "✅ Python env sẵn sàng: \$(${PY} --version)"
                """
            }
        }

        // ─────────────────────────────────────────────────
        // STAGE 1: Data Pipeline — Snorkel label → nlu.yml
        // ─────────────────────────────────────────────────
        stage('1. Data Pipeline') {
            steps {
                echo "🧹 Chạy csv_to_rasa.py để append data vào nlu.yml..."
                sh """
                    set -e
                    ${PY} data/csv_to_rasa.py
                """
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
                        export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}"
                        export MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT}"
                        export CHATBOT_S3_BUCKET="${CHATBOT_S3_BUCKET}"
                        export CHATBOT_S3_MODEL_KEY="${CHATBOT_S3_MODEL_KEY}"

                        ${PY} scripts/train_mlflow.py
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
                        credentialsId: 'github-credentials',
                        usernameVariable: 'GH_USER',
                        passwordVariable: 'GH_TOKEN'
                    )]) {
                        sh """
                            set -e

                            # Clone nếu chưa có, pull nếu đã có
                            if [ -d "${K8S_MANIFESTS_DIR}/.git" ]; then
                                cd "${K8S_MANIFESTS_DIR}"
                                git pull origin main
                            else
                                rm -rf "${K8S_MANIFESTS_DIR}"
                                git clone https://\${GH_USER}:\${GH_TOKEN}@github.com/minhnhatuit734/k8s-manifests.git "${K8S_MANIFESTS_DIR}"
                                cd "${K8S_MANIFESTS_DIR}"
                            fi

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

                            git push https://\${GH_USER}:\${GH_TOKEN}@github.com/minhnhatuit734/k8s-manifests.git main
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