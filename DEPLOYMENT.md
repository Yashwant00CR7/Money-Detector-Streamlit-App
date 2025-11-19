# Deployment Guide

## Deployment Options

This Streamlit Currency Detection app can be deployed on various platforms. Below are the most common options:

---

## 1. Streamlit Community Cloud (Recommended - Free)

### Prerequisites
- GitHub repository with your code
- Streamlit Community Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Steps
1. **Push your code to GitHub** (see Git LFS setup below)
2. **Sign in** to [Streamlit Community Cloud](https://share.streamlit.io)
3. **Click "New app"**
4. **Select your repository** and branch
5. **Set main file path**: `app.py`
6. **Deploy!**

### Configuration
Create `.streamlit/config.toml` (optional):
```toml
[server]
maxUploadSize = 200

[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

---

## 2. Hugging Face Spaces

### Steps
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select **Streamlit** as the SDK
3. Clone the Space repository
4. Copy your files to the Space
5. Push to Hugging Face

### Configuration
Create `README.md` in the Space:
```yaml
---
title: Currency Detection
emoji: 💵
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.44.1
app_file: app.py
pinned: false
---
```

---

## 3. Docker Deployment

### Dockerfile
Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run
```bash
# Build image
docker build -t currency-detector .

# Run container
docker run -p 8501:8501 currency-detector
```

---

## 4. Heroku Deployment

### Prerequisites
- Heroku account
- Heroku CLI installed

### Setup Files

**`setup.sh`**:
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

**`Procfile`**:
```
web: sh setup.sh && streamlit run app.py
```

### Deploy
```bash
heroku login
heroku create your-app-name
git push heroku main
```

---

## Git LFS Setup (For Model Files)

### Install Git LFS
```bash
# Windows (using Git for Windows)
git lfs install

# macOS
brew install git-lfs
git lfs install

# Linux
sudo apt-get install git-lfs
git lfs install
```

### Track Model Files
```bash
# Already configured in .gitattributes
# Verify with:
cat .gitattributes

# Track files
git lfs track "*.h5"
git lfs track "*.pt"
```

### Push to GitHub
```bash
# Initialize repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit with Git LFS models"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push
git branch -M main
git push -u origin main
```

### Verify LFS
```bash
# Check LFS files
git lfs ls-files

# Should show:
# Currency_Detection_model_with_DenseNet121_and_CentralFocusSpatialAttention.h5
# runs/detect/train4/weights/best.pt
```

---

## Environment Variables (if needed)

For sensitive data, use environment variables:

### Streamlit Cloud
1. Go to your app settings
2. Add secrets in the "Secrets" section
3. Access in code: `st.secrets["key"]`

### Local `.env` file
```env
MODEL_PATH=./Currency_Detection_model_with_DenseNet121_and_CentralFocusSpatialAttention.h5
YOLO_PATH=./runs/detect/train4/weights/best.pt
```

---

## Performance Optimization

### For Production
1. **Enable caching**: Already done with `@st.cache_resource`
2. **Optimize model loading**: Models load once and are cached
3. **Set resource limits**: Configure in deployment platform

### Monitoring
- Use Streamlit's built-in analytics
- Add custom logging if needed

---

## Troubleshooting

### Large File Issues
- **GitHub**: 100MB file size limit (use Git LFS)
- **Streamlit Cloud**: 1GB total repository size limit
- **Hugging Face**: 50GB limit per Space

### Memory Issues
- Reduce model batch size
- Use model quantization
- Deploy on platform with more RAM

### Slow Loading
- Ensure models are cached
- Use CDN for static assets
- Consider model optimization

---

## Security Considerations

1. **Never commit**:
   - API keys
   - Passwords
   - Private data

2. **Use `.gitignore`** for sensitive files

3. **Environment variables** for configuration

---

## Next Steps

1. ✅ Setup Git LFS
2. ✅ Push to GitHub
3. ✅ Deploy to Streamlit Cloud
4. ✅ Test deployment
5. ✅ Share your app!

---

## Support

For issues:
- Check [Streamlit docs](https://docs.streamlit.io)
- Visit [Streamlit forum](https://discuss.streamlit.io)
- Open an issue on GitHub
