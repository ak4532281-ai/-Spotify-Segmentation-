# -Spotify-Segmentation-
Spotify Genre Segmentation is a machine learning project that classifies songs into different genres using audio features like energy, danceability, tempo, and loudness. A Random Forest model is trained on Spotify data to predict genres accurately, supporting music recommendation systems.

The Spotify Songs Genre Segmentation project focuses on classifying songs into different music genres using machine learning techniques. Spotify provides a large dataset containing various audio features such as danceability, energy, loudness, tempo, valence, acousticness, and more. These features help in understanding the musical characteristics of each track.

In this project, the dataset is first cleaned by removing missing values and unnecessary text-based columns like artist name, track name, and album information. The target variable used for classification is the playlist_genre column, which represents the genre category of each song. Since machine learning models require numerical input, only numeric audio features are selected, and genre labels are encoded into numerical form.

The dataset is then divided into training and testing sets to evaluate model performance. Feature scaling is applied using StandardScaler to normalize the input values. A Random Forest Classifier is trained on the processed data because it provides high accuracy and handles multi-class classification effectively.

The trained model successfully predicts the genre of songs based on their audio attributes. This project demonstrates how machine learning can be applied in music recommendation systems, genre classification, and digital music analytics. Overall, it provides a practical approach to understanding and organizing music content automatically.
