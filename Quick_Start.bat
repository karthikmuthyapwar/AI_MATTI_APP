@echo off
echo =========================================
echo Starting Maati AI Recommendation System
echo =========================================
echo.
echo [1] Retraining AI Model on Latest Data...
cd backend
python train_model.py

echo.
echo [2] Launching Backend API...
start "Maati Backend API" cmd /k "python -m uvicorn main:app --reload --port 8000"

echo.
echo [3] Launching Frontend Interface...
cd ../frontend
start "Maati Frontend WebApp" cmd /k "npm run dev"

echo.
echo [4] Opening Dashboard in your browser...
:: Wait 3 seconds for the Vite development server to boot up
timeout /t 3 /nobreak >nul
start http://localhost:5173

echo.
echo Have a great day! Both servers are now running in background windows.
