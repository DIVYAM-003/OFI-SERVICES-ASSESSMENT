This project is a working prototype built for the NexGen Logistics Innovation Challenge.

The main goal was to help NexGen shift from being reactive to predictive by using their own data to spot delivery delays before they happen.

The Problem:-

After looking at the data, it was clear that a significant number of deliveries were being delayed. This hurts customer satisfaction (leading to bad ratings) and costs the company money. Right now, the team only finds out about a delay after it's already late.

My Solution:-

I built a two-in-one web application using Streamlit that any operations manager can use.

1. Performance Dashboard

A live dashboard that gives a high-level view of the company's performance. It answers simple questions like:

What's our overall on-time delivery rate?

Which carriers are performing the worst?

Are certain product categories more likely to be delayed?

2. Predictive Optimizer

This is the core tool. A manager can enter the details for a new order (like its origin, destination, carrier, and priority) and the tool will instantly provide a "Delay Risk Score."

This allows the team to be proactive. If an order has a "High Risk" score, they can immediately take action, like switching to a better carrier or alerting the customer in advance.

How It Works

The "brain" of the optimizer is a Random Forest machine learning model. I trained this model on all 7 of the company's datasets (like orders, delivery_performance, and routes_distance) so it could learn the specific patterns that lead to delays.

Technology Used

Python

Streamlit (for building the web app)

Pandas (for all the data handling and analysis)

Scikit-learn (for building the machine learning model)

Plotly (for the interactive charts)

How to Get it Running

Here are the steps to run this project on your own machine.

1. Set Up the Folder:

Make sure you have a main project folder (e.g., NexGen_Project).

Inside that folder, create a new folder named exactly data.

Place all 7 of the original CSV files (orders.csv, delivery_performance.csv, etc.) into that data folder.

2. Install the Libraries:

Open your terminal (like VS Code Terminal or Command Prompt) and navigate into your main project folder.

Run this command to install all the required libraries:

pip install -r requirements.txt


3. Train the Model (One-Time Step):

This is the most important step. In your terminal, run the training script:

python train_model.py


This will read all the CSVs, train the AI "brain," and save it as a file named delivery_model.pkl.

4. Run the App:

Once the model is trained, you can start the web app. Run this command in your terminal:

streamlit run app.py


Your web browser should open automatically, and you can start using the tool!
