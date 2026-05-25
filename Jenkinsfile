pipeline {

    agent { label 'chatbot_mlops_214' }

    environment {
        PYTHON      = "/home/thinh/Chatbot_tien/.venv/bin/python"
        VENV_BIN    = "/home/thinh/Chatbot_tien/.venv/bin"

        PROJECT_DIR = "${WORKSPACE}"
        MODEL_DIR   = "${WORKSPACE}/rasa_bot/models"

        LD_LIB      = "/home/thinh/miniconda3/envs/rasa/lib"

        MLFLOW_URI  = "http://127.0.0.1:5000"
        ENV_FILE    = "/home/thinh/Chatbot_tien/.env"

        QA_VENV     = "/home/jenkins/.cache/chatbot-qa-tools"
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 90, unit: 'MINUTES')
        parallelsAlwaysFailFast()
    }

    stages {

        stage('1. Verify Environment') {
    steps {
        echo "Verify Jenkins agent, Python, Rasa and MLflow"

        sh '''
            set -e

            export PATH="${VENV_BIN}:$PATH"
            export LD_LIBRARY_PATH="${LD_LIB}:${LD_LIBRARY_PATH:-}"

            echo "Workspace: ${PROJECT_DIR}"

            echo "Python version:"
            "${PYTHON}" --version

            echo "Check Python packages:"
            "${PYTHON}" -c "import rasa; print('Rasa:', rasa.__version__)"
            "${PYTHON}" -c "import mlflow; print('MLflow:', mlflow.__version__)"

            echo "Check Rasa CLI:"
            which rasa
            rasa --version

            echo "Copy .env to workspace"
            cp "${ENV_FILE}" "${PROJECT_DIR}/.env" || true

            echo "Check MLflow server"
            if curl -fsS "${MLFLOW_URI}" >/dev/null 2>&1; then
                echo "MLflow server is running at ${MLFLOW_URI}"
            else
                echo "MLflow server is not running. Starting MLflow server."

                mkdir -p /home/jenkins/mlflow/artifacts

                nohup "${PYTHON}" -m mlflow server \
                    --host 0.0.0.0 \
                    --port 5000 \
                    --backend-store-uri sqlite:////home/jenkins/mlflow/mlflow.db \
                    --default-artifact-root /home/jenkins/mlflow/artifacts \
                    > /home/jenkins/mlflow/mlflow-server.log 2>&1 &

                sleep 10

                curl -fsS "${MLFLOW_URI}" >/dev/null
                echo "MLflow server started successfully"
            fi
        '''
    }
}

        stage('2. Prepare QA Tools') {
            steps {
                echo "Prepare Python QA tools"

                sh '''
                    set -e

                    mkdir -p "$(dirname "${QA_VENV}")"

                    if [ ! -x "${QA_VENV}/bin/python" ]; then
                        echo "Create QA virtual environment"
                        "${PYTHON}" -m venv "${QA_VENV}"
                    fi

                    if [ ! -x "${QA_VENV}/bin/ruff" ] || \
                       [ ! -x "${QA_VENV}/bin/bandit" ] || \
                       [ ! -x "${QA_VENV}/bin/pip-audit" ]; then

                        echo "Install missing QA tools"
                        "${QA_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
                        "${QA_VENV}/bin/python" -m pip install --upgrade ruff bandit pip-audit
                    else
                        echo "QA tools already exist"
                    fi

                    "${QA_VENV}/bin/ruff" --version
                    "${QA_VENV}/bin/bandit" --version
                    "${QA_VENV}/bin/pip-audit" --version
                '''
            }
        }

        stage('3. Quality Checks') {
            parallel {

                stage('3.1 Validate Python Code') {
                    steps {
                        echo "Validate Python syntax and critical lint"

                        sh '''
                            set -e

                            cd "${PROJECT_DIR}"
                            mkdir -p reports

                            echo "Compile Python files"
                            "${PYTHON}" -m compileall -q data scripts rasa_bot

                            echo "Run ruff critical checks"
                            "${QA_VENV}/bin/ruff" check data scripts rasa_bot \
                                --select E9,F63,F7,F82 \
                                --output-format=github \
                                | tee reports/ruff-critical.txt

                            echo "Python validation completed"
                        '''
                    }
                }

                stage('3.2 Scan Libraries') {
                    steps {
                        echo "Check Python dependencies and known vulnerabilities"

                        sh '''
                            set -e

                            cd "${PROJECT_DIR}"
                            mkdir -p reports

                            echo "Export installed packages"
                            "${PYTHON}" -m pip freeze > reports/pip-freeze.txt

                            echo "Check dependency conflicts"
                            "${PYTHON}" -m pip check | tee reports/pip-check.txt

                            echo "Run pip-audit vulnerability scan"
                            if [ -f "requirements.txt" ]; then
                                timeout 300 "${QA_VENV}/bin/pip-audit" \
                                    -r requirements.txt \
                                    > reports/pip-audit.txt 2>&1 || true
                            else
                                timeout 300 "${QA_VENV}/bin/pip-audit" \
                                    > reports/pip-audit.txt 2>&1 || true
                            fi

                            echo "Library scan completed"
                        '''
                    }
                }

                stage('3.3 Scan Code Security and Malware') {
                    steps {
                        echo "Scan Python security issues and malware"

                        sh '''
                            set -e

                            cd "${PROJECT_DIR}"
                            mkdir -p reports

                            echo "Run Bandit security scan"
                            "${QA_VENV}/bin/bandit" \
                                -r data scripts rasa_bot \
                                -x "**/__pycache__/**,**/.venv/**,**/tests/**" \
                                -ll -ii \
                                -f json \
                                -o reports/bandit.json || true

                            echo "Run ClamAV malware scan if available"
                            if command -v clamscan >/dev/null 2>&1; then
                                clamscan -r \
                                    --infected \
                                    --exclude-dir=".git" \
                                    --exclude-dir=".venv" \
                                    --exclude-dir="__pycache__" \
                                    . > reports/clamscan.txt
                            else
                                echo "clamscan is not installed. Malware scan skipped." > reports/clamscan.txt
                            fi

                            echo "Security and malware scan completed"
                        '''
                    }
                }
            }
        }

        stage('4. Data Pipeline') {
            steps {
                echo "Convert CSV data to Rasa format"

                sh '''
                    set -e

                    cd "${PROJECT_DIR}"

                    export PATH="${VENV_BIN}:$PATH"
                    export LD_LIBRARY_PATH="${LD_LIB}:${LD_LIBRARY_PATH:-}"

                    "${PYTHON}" data/csv_to_rasa.py

                    echo "Sync generated NLU file to rasa_bot/data"
                    if [ -f "data/nlu_test.yml" ]; then
                        cp data/nlu_test.yml rasa_bot/data/nlu_test.yml
                        echo "Copied data/nlu_test.yml to rasa_bot/data/nlu_test.yml"
                    else
                        echo "data/nlu_test.yml not found. Skip copy."
                    fi

                    echo "Data pipeline completed"
                '''
            }
        }

        stage('5. Train Model') {
            steps {
                echo "Train Rasa model and log metrics to MLflow"

                sh '''
                    set -e

                    cd "${PROJECT_DIR}"

                    export PATH="${VENV_BIN}:$PATH"
                    export LD_LIBRARY_PATH="${LD_LIB}:${LD_LIBRARY_PATH:-}"
                    export MLFLOW_TRACKING_URI="${MLFLOW_URI}"

                    echo "Check Rasa CLI before training"
                    which rasa
                    rasa --version

                    "${PYTHON}" scripts/train_mlflow.py

                    echo "Model training completed"
                '''
            }
        }

        stage('6. Check Model Artifact') {
            steps {
                echo "Check generated model artifact"

                sh '''
                    set -e

                    echo "Model directory: ${MODEL_DIR}"
                    ls -lah "${MODEL_DIR}" || true

                    LATEST_MODEL=$(ls -t "${MODEL_DIR}"/*.tar.gz 2>/dev/null | head -n 1 || true)

                    if [ -z "$LATEST_MODEL" ]; then
                        echo "No model .tar.gz found in ${MODEL_DIR}"
                        exit 1
                    fi

                    echo "Latest model: $LATEST_MODEL"
                    echo "$LATEST_MODEL" > latest_model_path.txt
                '''
            }
        }

        stage('7. Human Approval') {
            steps {
                script {
                    def latestModel = sh(
                        script: 'cat latest_model_path.txt',
                        returnStdout: true
                    ).trim()

                    echo "Model trained: ${latestModel}"
                    echo "Check MLflow metrics before deployment."

                    def decision = input(
                        id: 'DeployGate',
                        message: "Model: ${latestModel}\nCheck MLflow metrics. Deploy?",
                        ok: 'Submit',
                        parameters: [
                            choice(
                                name: 'DECISION',
                                choices: ['deploy', 'reject'],
                                description: 'deploy = deploy model, reject = stop pipeline'
                            )
                        ]
                    )

                    if (decision == 'reject') {
                        currentBuild.result = 'ABORTED'
                        error("Model rejected. Pipeline stopped.")
                    }

                    echo "Model approved. Continue deployment."
                }
            }
        }

        stage('8. Deploy Model') {
            steps {
                echo "Deploy model"

                sh '''
                    set -e

                    cd "${PROJECT_DIR}"

                    export PATH="${VENV_BIN}:$PATH"
                    export LD_LIBRARY_PATH="${LD_LIB}:${LD_LIBRARY_PATH:-}"

                    "${PYTHON}" scripts/deploy_model.py

                    echo "Deployment completed"
                '''
            }
        }
    }

    post {

        always {
            echo "Archive reports and artifacts"

            archiveArtifacts artifacts: 'latest_model_path.txt,error_log.txt,reports/**/*,rasa_bot/results/**/*,rasa_bot/models/*.tar.gz', allowEmptyArchive: true
        }

        success {
            echo "Pipeline completed successfully"
        }

        aborted {
            echo "Pipeline aborted or model rejected. Deployment skipped."
        }

        failure {
            echo "Pipeline failed. Check Console Output and archived reports."
        }
    }
}