pipeline {

    agent { label 'chatbot-mlops' }

    environment {
        PYTHON      = "/home/thinh/Chatbot_tien/.venv/bin/python"
        PROJECT_DIR = "${WORKSPACE}"
        MODEL_DIR   = "${WORKSPACE}/rasa_bot/models"
        LD_LIB      = "/home/thinh/miniconda3/envs/rasa/lib"
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 90, unit: 'MINUTES')
    }

    stages {

        stage('1. Verify Environment') {
            steps {
                sh """
                    set -e
                    ${PYTHON} --version
                    ${PYTHON} -c "import rasa; print('Rasa:', rasa.__version__)"
                    ${PYTHON} -c "import mlflow; print('MLflow:', mlflow.__version__)"
                    cp /home/thinh/Chatbot_tien/.env ${PROJECT_DIR}/.env || true
                """
            }
        }

        stage('2. Data Pipeline') {
            steps {
                echo "🔄 Convert CSV sang Rasa format..."
                sh """
                    set -e
                    cd "${PROJECT_DIR}"
                    export LD_LIBRARY_PATH="${LD_LIB}:\$LD_LIBRARY_PATH"
                    ${PYTHON} data/csv_to_rasa.py
                """
            }
        }

        stage('3. Train Model') {
            steps {
                echo "🚀 Train Rasa + log MLflow..."
                sh """
                    set -e
                    cd "${PROJECT_DIR}"
                    export LD_LIBRARY_PATH="${LD_LIB}:\$LD_LIBRARY_PATH"
                    ${PYTHON} scripts/train_mlflow.py
                """
            }
        }

        stage('4. Check Model Artifact') {
            steps {
                sh """
                    set -e
                    LATEST_MODEL=\$(ls -t "${MODEL_DIR}"/*.tar.gz 2>/dev/null | head -n 1 || true)
                    if [ -z "\$LATEST_MODEL" ]; then
                        echo "❌ Không tìm thấy model"
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

                    def decision = input(
                        id: 'DeployGate',
                        message: "Model: ${latestModel}\nKiểm tra MLflow xong. Deploy không?",
                        ok: 'Submit',
                        parameters: [
                            choice(
                                name: 'DECISION',
                                choices: ['deploy', 'reject'],
                                description: 'deploy = triển khai, reject = dừng'
                            )
                        ]
                    )

                    if (decision == 'reject') {
                        currentBuild.result = 'ABORTED'
                        error("🛑 Model bị reject.")
                    }
                    echo "✅ Approved."
                }
            }
        }

        stage('6. Deploy Model') {
            steps {
                sh """
                    set -e
                    cd "${PROJECT_DIR}"
                    export LD_LIBRARY_PATH="${LD_LIB}:\$LD_LIBRARY_PATH"
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
            echo "🔥 Pipeline thất bại."
        }
    }
}