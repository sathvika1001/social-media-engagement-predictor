# Social Media Engagement Predictor

A machine learning web app that predicts the engagement (likes + comments + shares) of a social media post and shows average engagement patterns from past posts.

## Tech Stack
- **Model:** Python, scikit-learn (Random Forest Regressor)
- **Backend:** Flask REST API
- **Frontend:** React (Vite)

## Features
- Predict engagement for a post based on platform, post type, day, and hour
- Dashboard showing average engagement by post type, day, and hour

## Dataset
- 100 social media posts (Facebook, Instagram, Twitter)
- Engagement = likes + comments + shares
- Post hour is taken from the post timestamp

## How to Run

1. Install the packages:
   ```bash
   pip install flask pandas numpy joblib scikit-learn
   ```
2. Train the model (optional, a trained model is included):
   ```bash
   python train.py
   ```
3. Start the Flask API on port 5000:
   ```bash
   python app.py
   ```
4. In a second terminal, start the React app on port 5173:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
5. Open http://localhost:5173

## Results and Limitations
- The model's R² score is about 0.08 on the test split, so predictions are rough estimates.
- The dataset is small (100 posts) and does not include content quality or audience size, which strongly affect engagement.
- The dashboard averages are more reliable than individual predictions.

## Future Work
- Collect more posts to improve the model
- Add features such as caption length, hashtags, and follower count
- Serve the React build directly from Flask so only one server is needed
