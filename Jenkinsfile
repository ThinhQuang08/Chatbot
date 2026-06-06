pipeline {
    agent any
    environment {
        PYTHON_CMD = 'python'
        PROJECT_DIR = "${WORKSPACE}"
        MODEL_DIR   = "${WORKSPACE}/rasa_bot/models"
        // DockerHub image name (Thay bằng username DockerHub của bạn nếu cần)
        DOCKER_IMAGE = "mnhat1/chatbot-rasa"
        // k8s-manifests repo để update image tag
        K8S_MANIFESTS_REPO = "https://github.com/minhnhatuit734/k8s-manifests.git"
        K8S_MANIFESTS_DIR  = "/workspace/k8s-manifests"
        // AWS S3 bucket (non-secret config)
        CHATBOT_S3_BUCKET    = "kltn-chatbot-artifacts-dev"
        CHATBOT_S3_MODEL_KEY = "models/latest_model.tar.gz"
        AWS_DEFAULT_REGION   = "ap-southeast-1"
    }
    stages {
        stage('1. Data Pipeline') {
            steps {
                echo "🧹 Làm sạch, gán nhãn bằng Snorkel và Validate..."
                sh """
                    cd /workspace
                    ${PYTHON_CMD} data/csv_to_rasa.py
                """
            }
        }
        stage('2. Train Model') {
            steps {
                echo "🚀 Train Rasa và log metrics lên MLflow..."
                sh """
                    cd /workspace
                    ${PYTHON_CMD} scripts/train_mlflow.py
                """
            }
        }
        stage('3. Human Approval') {
            steps {
                script {
                    echo "🔔 Chờ review metrics trên MLflow..."
                    def userInput = input(
                        id: 'DeployGate',
                        message: 'Metrics đã có trên MLflow. Quyết định deploy?',
                        ok: 'Deploy',
                        parameters: [
                            choice(
                                name: 'DECISION',
                                choices: ['deploy', 'reject'],
                                description: 'deploy = deploy model, reject = stop pipeline'
                            )
                        ]
                    )
                    if (userInput == 'nhu_cc_xoa') {
                        error("🛑 Model bị reject")
                    }
                    echo "✅ Model được duyệt — tiến hành deploy"
                }
            }
        }
        stage('4. Upload Model → S3') {
            steps {
                echo "☁️ Upload model artifact lên S3..."
                withCredentials([
                    string(credentialsId: 'aws-access-key-id',     variable: 'AWS_ACCESS_KEY_ID'),
                    string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
                ]) {
                    sh """
                        cd /workspace
                        ${PYTHON_CMD} scripts/deploy_model.py
                    """
                }
            }
        }
        stage('5. Build & Push Docker Image') {
            steps {
                script {
                    // Tag = dev-<BUILD_NUMBER>-<git short SHA>
                    def gitSha = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    env.IMAGE_TAG = "dev-${BUILD_NUMBER}-${gitSha}"
                    env.FULL_IMAGE = "${DOCKER_IMAGE}:${env.IMAGE_TAG}"
                    echo "🐳 Building image: ${env.FULL_IMAGE}"
                    // Download model từ S3 vào rasa_bot/models/ trước khi build
                    withCredentials([
                        string(credentialsId: 'aws-access-key-id',     variable: 'AWS_ACCESS_KEY_ID'),
                        string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
                    ]) {
                        sh """
                            mkdir -p /workspace/rasa_bot/models
                            aws s3 cp s3://${CHATBOT_S3_BUCKET}/${CHATBOT_S3_MODEL_KEY} \
                                /workspace/rasa_bot/models/latest_model.tar.gz \
                                --region ${AWS_DEFAULT_REGION}
                        """
                    }
                    withCredentials([usernamePassword(
                        credentialsId: 'dockerhub-credentials',
                        usernameVariable: 'DOCKER_USER',
                        passwordVariable: 'DOCKER_PASS'
                    )]) {
                        sh """
                            echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                            docker build -f Dockerfile.prod -t ${env.FULL_IMAGE} /workspace
                            docker push ${env.FULL_IMAGE}
                            docker rmi ${env.FULL_IMAGE} || true
                        """
                    }
                    echo "✅ Image pushed: ${env.FULL_IMAGE}"
                }
            }
        }
        stage('6. Update k8s-manifests Image Tag') {
            steps {
                script {
                    echo "📝 Cập nhật image tag trong k8s-manifests..."
                    withCredentials([usernamePassword(
                        credentialsId: 'github-credentials',
                        usernameVariable: 'GH_USER',
                        passwordVariable: 'GH_TOKEN'
                    )]) {
                        sh """
                            set -e
                            # Clone k8s-manifests nếu chưa có
                            if [ -d "${K8S_MANIFESTS_DIR}" ]; then
                                cd ${K8S_MANIFESTS_DIR} && git pull
                            else
                                git clone https://${GH_USER}:${GH_TOKEN}@github.com/minhnhatuit734/k8s-manifests.git ${K8S_MANIFESTS_DIR}
                            fi
                            cd ${K8S_MANIFESTS_DIR}
                            # Update newTag dùng kustomize edit
                            cd overlays/dev
                            kustomize edit set image mnhat1/chatbot-rasa=${env.FULL_IMAGE}
                            cd ${K8S_MANIFESTS_DIR}
                            git config user.email "jenkins@kltn.local"
                            git config user.name "Jenkins Bot"
                            git add overlays/dev/kustomization.yaml
                            git commit -m "chatbot: update rasa image to ${env.IMAGE_TAG} [skip ci]" || echo "No changes to commit"
                            git push https://${GH_USER}:${GH_TOKEN}@github.com/minhnhatuit734/k8s-manifests.git main
                        """
                    }
                    echo "🚀 ArgoCD sẽ sync image tag mới vào EKS"
                }
            }
        }
    }
    post {
        success {
            echo "🎉 PIPELINE HOÀN TẤT! Image: ${env.FULL_IMAGE ?: 'N/A'}"
        }
        aborted {
            echo "⚠️ Pipeline bị hủy"
            sh "rm -f /workspace/rasa_bot/models/latest_model.tar.gz || true"
        }
        failure {
            echo "🔥 Pipeline thất bại"
            sh "rm -f /workspace/rasa_bot/models/latest_model.tar.gz || true"
            echo "🗑️ Đã cleanup model file"
        }
    }
}