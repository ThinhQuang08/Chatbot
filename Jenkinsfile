pipeline {

    agent { label 'chatbot-mlops-214' }

    environment {
        PYTHON_CMD = "/home/thinh/Chatbot_tien/.venv/bin/python"

        // Workspace thật của Jenkins
        PROJECT_DIR = "${WORKSPACE}"

        // File .env gốc đang nằm ở thư mục runtime cũ
        ENV_FILE = "/home/thinh/Chatbot_tien/.env"

        // Thư mục model sau khi train
        MODEL_DIR = "${WORKSPACE}/rasa_bot/models"

        // Library path cho Rasa/TensorFlow nếu môi trường cần
        LD_LIB = "/home/thinh/miniconda3/envs/rasa/lib"
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 90, unit: 'MINUTES')
    }

    stages {

        stage('1. Verify Environment') {
            steps {
                echo "✅ Kiểm tra môi trường Jenkins Agent + Python..."

                sh '''
                    set -e

                    echo "Workspace: ${PROJECT_DIR}"
                    cd "${PROJECT_DIR}"

                    echo "Copy .env vào workspace..."
                    if [ -f "${ENV_FILE}" ]; then
                        cp "${ENV_FILE}" .env
                        echo "✅ Đã copy .env"
                    else
                        echo "⚠️ Không tìm thấy ${ENV_FILE}, bỏ qua copy .env"
                    fi

                    echo "Python version:"
                    ${PYTHON_CMD} --version

                    echo "Kiểm tra package chính:"
                    ${PYTHON_CMD} -c "import rasa; print('Rasa:', rasa.__version__)"
                    ${PYTHON_CMD} -c "import mlflow; print('MLflow:', mlflow.__version__)"
                    ${PYTHON_CMD} -c "import sentence_transformers; print('SentenceTransformers:', sentence_transformers.__version__)"

                    echo "Disk:"
                    df -h

                    echo "Memory:"
                    free -h || true
                '''
            }
        }

        stage('2. Code & Library Check') {
            parallel {

                stage('2.1 Validate Python Code') {
                    steps {
                        echo "🐍 Validate Python code..."

                        sh '''
                            set -e
                            cd "${PROJECT_DIR}"

                            echo "Kiểm tra syntax Python..."
                            ${PYTHON_CMD} -m compileall -q data scripts rasa_bot

                            echo "Nếu có ruff thì chạy ruff check..."
                            if command -v ruff >/dev/null 2>&1; then
                                ruff check data scripts rasa_bot --select E9,F63,F7,F82
                            else
                                echo "⚠️ ruff chưa được cài, bỏ qua ruff check"
                            fi

                            echo "✅ Python code hợp lệ"
                        '''
                    }
                }

                stage('2.2 Check Libraries') {
                    steps {
                        echo "📦 Kiểm tra thư viện Python..."

                        sh '''
                            set -e
                            cd "${PROJECT_DIR}"

                            mkdir -p reports

                            echo "Lưu danh sách thư viện..."
                            ${PYTHON_CMD} -m pip freeze > reports/pip-freeze.txt

                            echo "Kiểm tra dependency conflict bằng pip check..."
                            ${PYTHON_CMD} -m pip check | tee reports/pip-check.txt

                            echo "Nếu có pip-audit thì scan vulnerability..."
                            if command -v pip-audit >/dev/null 2>&1; then
                                pip-audit -r requirements.txt > reports/pip-audit.txt || true
                                echo "⚠️ pip-audit chỉ tạo report, không chặn pipeline ở giai đoạn demo"
                            else
                                echo "⚠️ pip-audit chưa được cài, bỏ qua vulnerability scan" | tee reports/pip-audit.txt
                            fi

                            echo "✅ Check libraries hoàn tất"
                        '''
                    }
                }

                stage('2.3 Malware Scan') {
                    steps {
                        echo "🛡️ Scan mã độc nếu máy có ClamAV..."

                        sh '''
                            set -e
                            cd "${PROJECT_DIR}"

                            mkdir -p reports

                            if command -v clamscan >/dev/null 2>&1; then
                                clamscan -r \
                                    --infected \
                                    --exclude-dir=".git" \
                                    --exclude-dir=".venv" \
                                    --exclude-dir="__pycache__" \
                                    . | tee reports/clamscan.txt
                            else
                                echo "⚠️ clamscan chưa được cài, bỏ qua malware scan" | tee reports/clamscan.txt
                            fi

                            echo "✅ Malware scan hoàn tất"
                        '''
                    }
                }
            }
        }

        stage('3. Data Pipeline') {
            steps {
                echo "🧹 Đang xử lý dữ liệu, gán nhãn và validate..."

                sh '''
                    set -e
                    cd "${PROJECT_DIR}"

                    export LD_LIBRARY_PATH="${LD_LIB}:${LD_LIBRARY_PATH}"
                    
                    ${PYTHON_CMD} data/preprocess_data.py
                    ${PYTHON_CMD} data/auto_label_snorkel.py
                    ${PYTHON_CMD} data/split_confidence.py
                    ${PYTHON_CMD} data/validate_cleanlab.py
                    ${PYTHON_CMD} data/csv_to_rasa.py

                    echo "✅ Data pipeline hoàn tất"
                '''
            }
        }

        stage('4. Train Model') {
            steps {
                echo "🚀 Đang huấn luyện Rasa và lưu metrics lên MLflow..."

                sh '''
                    set -e
                    cd "${PROJECT_DIR}"

                    export LD_LIBRARY_PATH="${LD_LIB}:${LD_LIBRARY_PATH}"

                    ${PYTHON_CMD} scripts/train_mlflow.py

                    echo "✅ Train model hoàn tất"
                '''
            }
        }

        stage('5. Check Model Artifact') {
            steps {
                echo "🔎 Kiểm tra model artifact..."

                sh '''
                    set -e

                    echo "Model directory: ${MODEL_DIR}"
                    ls -lah "${MODEL_DIR}" || true

                    LATEST_MODEL=$(ls -t "${MODEL_DIR}"/*.tar.gz 2>/dev/null | head -n 1 || true)

                    if [ -z "${LATEST_MODEL}" ]; then
                        echo "❌ Không tìm thấy model .tar.gz"
                        exit 1
                    fi

                    echo "✅ Model mới nhất: ${LATEST_MODEL}"
                    echo "${LATEST_MODEL}" > latest_model_path.txt
                '''
            }
        }

        stage('6. Human Approval') {
            steps {
                script {
                    def latestModel = sh(
                        script: 'cat latest_model_path.txt',
                        returnStdout: true
                    ).trim()

                    echo "🔔 Model đã train xong: ${latestModel}"
                    echo "📊 Vào MLflow kiểm tra thông số trước khi deploy."

                    def userInput = input(
                        id: 'DeployGate',
                        message: "Thông số mô hình đã có trên MLflow. Có deploy model này không?\n${latestModel}",
                        ok: 'Submit',
                        parameters: [
                            choice(
                                name: 'DECISION',
                                choices: ['deploy', 'reject'],
                                description: 'deploy = triển khai model, reject = hủy pipeline'
                            )
                        ]
                    )

                    if (userInput == 'reject') {
                        currentBuild.result = 'ABORTED'
                        error("🛑 Model bị reject. Không deploy.")
                    }

                    echo "✅ Model được duyệt"
                }
            }
        }

        stage('7. Deploy to MinIO & Rasa') {
            steps {
                echo "☁️ Đang deploy model..."

                sh '''
                    set -e
                    cd "${PROJECT_DIR}"

                    export LD_LIBRARY_PATH="${LD_LIB}:${LD_LIBRARY_PATH}"

                    ${PYTHON_CMD} scripts/deploy_model.py

                    echo "✅ Deploy hoàn tất"
                '''
            }
        }
    }

    post {

        always {
            echo "📦 Lưu report nếu có..."

            archiveArtifacts artifacts: '''
                latest_model_path.txt,
                reports/**/*
            ''', allowEmptyArchive: true
        }

        success {
            echo "🎉 PIPELINE HOÀN TẤT!"
        }

        aborted {
            echo "⚠️ Pipeline bị hủy hoặc model bị reject"

            sh '''
                rm -f "${MODEL_DIR}"/*.tar.gz || true
            '''

            echo "🗑️ Đã cleanup model tạm"
        }

        failure {
            echo "🔥 Pipeline thất bại"

            sh '''
                rm -f "${MODEL_DIR}"/*.tar.gz || true
            '''

            echo "🗑️ Đã cleanup model tạm"
        }
    }
}