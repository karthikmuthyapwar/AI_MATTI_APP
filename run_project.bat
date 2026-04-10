@echo off
echo =========================================
echo Starting Maati Web application...
echo =========================================

echo Installing Backend Dependencies...
cd backend
pip install -r requirements.txt

echo Training Machine Learning Model...
python train_model.py

echo Starting Backend API Server...
start cmd /k "python -m uvicorn main:app --reload"

cd ../frontend
echo Installing Frontend Dependencies...
call npm install

echo Starting React Development Server...
start cmd /k "npm run dev"

echo Done! The UI should automatically reflect in your browser or at http://localhost:5173
