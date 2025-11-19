@echo off
REM Git Setup Script for Currency Detection App
REM This script initializes git, configures Git LFS, and pushes to GitHub

echo ========================================
echo Git Setup for Currency Detection App
echo ========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from https://git-scm.com/
    pause
    exit /b 1
)

REM Check if Git LFS is installed
git lfs version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git LFS is not installed
    echo Please install Git LFS from https://git-lfs.github.com/
    pause
    exit /b 1
)

echo [1/7] Installing Git LFS hooks...
git lfs install
if errorlevel 1 (
    echo WARNING: Git LFS install failed, but continuing...
)

echo.
echo [2/7] Initializing git repository...
if not exist .git (
    git init
    echo Repository initialized
) else (
    echo Repository already initialized
)

echo.
echo [3/7] Configuring Git LFS for model files...
REM .gitattributes already created, just verify
if exist .gitattributes (
    echo Git LFS configuration found in .gitattributes
) else (
    echo WARNING: .gitattributes not found
)

echo.
echo [4/7] Adding remote repository...
git remote remove origin >nul 2>&1
git remote add origin git@github.com:Yashwant00CR7/Money-Detector-Streamlit-App.git
echo Remote added: git@github.com:Yashwant00CR7/Money-Detector-Streamlit-App.git

echo.
echo [5/7] Adding all files to git...
git add .
echo Files staged for commit

echo.
echo [6/7] Creating initial commit...
git commit -m "Initial commit: Streamlit currency detection app with Git LFS models"
if errorlevel 1 (
    echo WARNING: Commit failed or nothing to commit
)

echo.
echo [7/7] Pushing to GitHub...
git branch -M main
git push -u origin main --force
if errorlevel 1 (
    echo.
    echo ERROR: Push failed!
    echo.
    echo Possible reasons:
    echo 1. SSH key not configured - run: ssh -T git@github.com
    echo 2. Repository doesn't exist on GitHub
    echo 3. No permission to push
    echo.
    echo To fix SSH issues:
    echo 1. Generate SSH key: ssh-keygen -t ed25519 -C "your_email@example.com"
    echo 2. Add to GitHub: https://github.com/settings/keys
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS! Repository pushed to GitHub
echo ========================================
echo.
echo Repository URL: https://github.com/Yashwant00CR7/Money-Detector-Streamlit-App
echo.
echo Next steps:
echo 1. Visit your repository on GitHub
echo 2. Verify model files are tracked by Git LFS
echo 3. Deploy to Streamlit Cloud (see DEPLOYMENT.md)
echo.
echo To verify Git LFS files:
echo   git lfs ls-files
echo.
pause
