pipeline {

    agent any

    environment {
        PYTHON_CMD = 'python'

        // workspace thật của Jenkins
        PROJECT_DIR = "${WORKSPACE}"

        MODEL_DIR = "${WORKSPACE}/rasa_bot/models"
    }

    stages {

        stage('1. Data Pipeline') {
            steps {
                echo "🧹 Đang làm sạch, gán nhãn bằng Snorkel và Validate..."

                sh """
                    cd /workspace
                    ${PYTHON_CMD} data/csv_to_rasa.py
                """
            }
        }

        stage('2. Train Model') {
            steps {
                echo "🚀 Đang huấn luyện Rasa và lưu metrics lên MLflow..."

                sh """
                    cd /workspace
                    ${PYTHON_CMD} scripts/train_mlflow.py
                """
            }
        }

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

        stage('4. Deploy to MinIO & Rasa') {
            steps {

                echo "☁️ Đang deploy model..."

                sh """
                    cd /workspace
                    ${PYTHON_CMD} scripts/deploy_model.py
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

            sh """
                rm -f /workspace/rasa_bot/models/*.tar.gz || true
            """
        }

        failure {

            echo "🔥 Pipeline thất bại"

            sh """
                rm -f /workspace/rasa_bot/models/*.tar.gz || true
            """

            echo "🗑️ Đã cleanup"
        }
    }
}