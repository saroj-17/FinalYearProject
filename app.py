import os
from pyexpat import features, model
from turtle import pd
import pandas as pd
from flask import Flask, render_template, request, redirect, session, url_for, flash
from flask import Flask, render_template, request
import stripe
from wtforms.validators import InputRequired, Length, Email
from flask_bootstrap import Bootstrap
from flask_cors import cross_origin
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_login import UserMixin
from sqlalchemy.orm import relationship
from datetime import datetime
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import joblib  # Use joblib for loading the model
from dotenv import load_dotenv
import numpy as np
from model import SimpleRandomForestRegressor


app = Flask(__name__)

stripe_keys = {
        "secret_key": os.environ["STRIPE_SECRET_KEY"],
        "publishable_key": os.environ["STRIPE_PUBLISHABLE_KEY"],
    }
stripe.api_key = stripe_keys["secret_key"]
stripe.api_key = stripe_keys["publishable_key"]

app.config['SECRET_KEY'] = 'hello1234'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost/Login'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
Bootstrap(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True,autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    contact =db.Column(db.String(120),unique=True, nullable=False)  

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=80)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=6, max=80)])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=80)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    email = StringField('Email', validators=[InputRequired(), Email()])
    contact = StringField('Contact', validators=[InputRequired(), Length(min=10, max=10, message='Contact number must be exactly 10 digits long')])
    submit = SubmitField('Register')

class ShowDetail(db.Model):
    __tablename__ = 'user_show'
    __table_args__ = {'schema': 'login'}

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    arrival_time = db.Column(db.DateTime, nullable=False)
    source = db.Column(db.String(255), nullable=False)
    destination = db.Column(db.String(255), nullable=False)
    stop = db.Column(db.String(255), nullable=False)  # Make sure 'stop' column is defined
    airline = db.Column(db.String(255), nullable=False)
    price = db.Column(db.Float, nullable=False)
    current_datetime = db.Column(db.DateTime, nullable=False)
    country = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f"ShowDetail(id={self.id}, user_id={self.user_id}, arrival_time={self.arrival_time}, source={self.source}, destination={self.destination}, stop={self.stop}, airline={self.airline}, price={self.price}, current_datetime={self.current_datetime}, country={self.country})"

load_dotenv() 

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

model = joblib.load('Flight_RF_model.pkl')
@app.route('/')
def home():
    username = None
    if current_user.is_authenticated:
        username = current_user.username
    return render_template('home.html',username=username)


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.password == password:
            login_user(user)
            flash('Login successful', 'success')
            return redirect(url_for('home'))
        else:
            flash('User is not registered or invalid password', 'error')
            return render_template('login.html', form=form)
    
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))



@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
       username = form.username.data
       password = form.password.data
       email = form.email.data
       contact = form.contact.data
       user = User(username=username, password=password,email=email,contact=contact)
       db.session.add(user)
       db.session.commit()

       session['user_email']=email
       flash('Registration successful. Please log in.', 'success')
       return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/checkout',methods=['POST'])
@login_required
def checkout():
    if current_user.is_authenticated:
        username = current_user.username
    publishable_key = os.environ.get("STRIPE_PUBLISHABLE_KEY")
   
    predicted_price = session.get("predicted_price")
 
    predicted_price_in_cent = int(predicted_price*100)
    return render_template('checkout.html', key=publishable_key,predicted_price=predicted_price_in_cent,username=username)
 
from datetime import datetime
from flask import render_template

@app.route('/charge', methods=['POST'])
def charge():
    if current_user.is_authenticated:
     username = current_user.username
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
    predicted_price = session.get("predicted_price")
    predicted_price_float = float(predicted_price)
    amount = int(predicted_price_float*100)

    user_email = session.get("user_email")
    if not user_email:
        flash('User email not found. Please register again.', 'error')
        return redirect(url_for('register'))
    
    user = User.query.filter_by(email=user_email).first()
    if not user:
        flash('User not found. Please register again.', 'error')
        return redirect(url_for('register'))

    customer = stripe.Customer.create(
        email=user_email,
        source=request.form['stripeToken']
    )

    charge = stripe.Charge.create(
        customer=customer.id,
        amount=amount,
        currency='usd',
        description='Flight Fare'
    )
    try:
        # Store user information in user_show table
        user_show = ShowDetail(
            user_id=current_user.id,
            arrival_time=session['input_data']['arrival_time'],
            source=session['input_data']['source'],
            destination=session['input_data']['destination'],
            stop=session['input_data']['stops'],
            airline=session['input_data']['airline'],
            price=session['input_data']['price'],
            current_datetime=datetime.now(),
            country=session['input_data']['country']
        )
        db.session.add(user_show)
        db.session.commit()
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        # Handle the exception, e.g., rollback the session and display an error message
        db.session.rollback()
        flash('An error occurred while saving the record. Please try again.', 'error')
        return redirect(url_for('show'))

    return render_template('charge.html', amount=amount,username=username)

@app.route('/user_show_detail', methods=['GET', 'POST'])
@login_required
def user_show_detail():
    user_id = current_user.id
    user_show_details = ShowDetail.query.filter_by(user_id=user_id).all()
    input_data = session.get("input_data", None)

    if input_data:
        username = current_user.username if current_user.is_authenticated else "Guest"
        email = current_user.email
        contact = current_user.contact
    return render_template('user_show_details.html', user_show_details=user_show_details,username=username,email=email,contact=contact)

##for getting current time
def get_current_datetime():
    now = datetime.now()
    current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_datetime

@app.route("/predict", methods = ["GET", "POST"])
@login_required
@cross_origin()
def predict():
    if current_user.is_authenticated:
     username = current_user.username
    if request.method == "POST":
         
        form = LoginForm()
        username = form.username.data

        country = request.form.get("country")
        # Date_of_Journey
        date_dep = request.form["Dep_Time"]
        Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)
        Journey_year = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").year)
        # print("Journey Date : ",Journey_day, Journey_month)

        # Departure
        Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
        # print("Departure : ",Dep_hour, Dep_min)

        # Arrival
        date_arr = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
        # print("Arrival : ", Arrival_hour, Arrival_min)

        # Duration
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)
        # print("Duration : ", dur_hour, dur_min)

        # Total Stops
        Total_stops = int(request.form["stops"])
        # print(Total_stops)

      
    airline=request.form['airline']
    if(airline=='Jet Airways'):
            Jet_Airways = 1
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            Buddha_Air = 0
            Yeti_Airlines = 0
            Shree_Airlines = 0
            SpiceJet = 0
            Air_asia = 0
            Vistara = 0
            GoAir = 0
            
    elif (airline=='IndiGo'):
            Jet_Airways = 0
            IndiGo = 1
            Air_India = 0
            Multiple_carriers = 0
            Buddha_Air = 0
            Yeti_Airlines = 0
            Shree_Airlines = 0
            SpiceJet = 0
            Air_asia = 0
            Vistara = 0
            GoAir = 0

    elif (airline=='Air India'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 1
            Multiple_carriers = 0
            Buddha_Air = 0
            Yeti_Airlines = 0
            Shree_Airlines = 0
            SpiceJet = 0
            Air_asia = 0
            Vistara = 0
            GoAir = 0
            
    elif (airline=='Multiple carriers'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 1
            Buddha_Air = 0
            Yeti_Airlines = 0
            Shree_Airlines = 0
            SpiceJet = 0
            Air_asia = 0
            Vistara = 0
            GoAir = 0 
            
    elif (airline=='SpiceJet'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            Buddha_Air = 0
            Yeti_Airlines = 0
            Shree_Airlines = 0
            SpiceJet = 1
            Air_asia = 0
            Vistara = 0
            GoAir = 0

    elif (airline=='Air_asia'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            Buddha_Air = 0
            Yeti_Airlines = 0
            Shree_Airlines = 0
            SpiceJet = 0
            Air_asia = 1
            Vistara = 0
            GoAir = 0
  
    elif (airline=='Vistara'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            Buddha_Air = 0
            Yeti_Airlines = 0
            Shree_Airlines = 0
            SpiceJet = 0
            Air_asia = 0
            Vistara = 1
            GoAir = 0

    elif (airline=='GoAir'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            Buddha_Air = 0
            Yeti_Airlines = 0
            Shree_Airlines = 0
            SpiceJet = 0
            Air_asia = 0
            Vistara = 0
            GoAir = 1

    elif (airline=='Buddha Air'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            Buddha_Air = 1
            Yeti_Airlines = 0
            Shree_Airlines = 0
            SpiceJet = 0
            Air_asia = 0
            Vistara = 0
            GoAir = 0

    elif (airline=='Yeti Airlines'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            Buddha_Air = 0
            Yeti_Airlines = 1
            Shree_Airlines = 0
            SpiceJet = 0
            Air_asia = 0
            Vistara = 0
            GoAir = 0

    elif (airline=='Shree Airlines'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            Buddha_Air = 0
            Yeti_Airlines = 0
            Shree_Airlines = 1
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            
    else:
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Yeti_Airlines = 0
            Shree_Airlines = 0
            Buddha_Air = 0

       
    Source = request.form["Source"]
    if (Source == 'Delhi'):
            s_Delhi = 1
            s_Kolkata = 0
            s_Kathmandu = 0
            s_Banglore = 0
            s_Mumbai = 0
            s_Biratnagar = 0
            s_Janakpur = 0
            s_Pokhara = 0
            s_Chennai = 0
            s_Nepalgunj = 0
            s_Simara = 0
            s_Bhairahawa = 0
            s_Bharatpur = 0
            s_Dhangadi = 0
            s_Bhadrapur = 0
            s_Rajbiraj = 0
          
    elif (Source == 'Kolkata'):
            s_Delhi = 0
            s_Kolkata = 1
            s_Kathmandu = 0
            s_Banglore = 0
            s_Mumbai = 0
            s_Biratnagar = 0
            s_Janakpur = 0
            s_Pokhara = 0
            s_Chennai = 0
            s_Nepalgunj = 0
            s_Simara = 0
            s_Bhairahawa = 0
            s_Bharatpur = 0
            s_Dhangadi = 0
            s_Bhadrapur = 0
            s_Rajbiraj = 0

    elif (Source == 'Kathmandu'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Kathmandu = 1
            s_Banglore = 0
            s_Mumbai = 0
            s_Biratnagar = 0
            s_Janakpur = 0
            s_Pokhara = 0
            s_Chennai = 0
            s_Nepalgunj = 0
            s_Simara = 0
            s_Bhairahawa = 0
            s_Bharatpur = 0
            s_Dhangadi = 0
            s_Bhadrapur = 0
            s_Rajbiraj = 0

    elif (Source == 'Banglore'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Kathmandu = 0
            s_Banglore = 1
            s_Mumbai = 0
            s_Biratnagar = 0
            s_Janakpur = 0
            s_Pokhara = 0
            s_Chennai = 0
            s_Nepalgunj = 0
            s_Simara = 0
            s_Bhairahawa = 0
            s_Bharatpur = 0
            s_Dhangadi = 0
            s_Bhadrapur = 0
            s_Rajbiraj = 0

    elif (Source == 'Mumbai'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Kathmandu = 0
            s_Banglore = 0
            s_Mumbai = 1
            s_Biratnagar = 0
            s_Janakpur = 0
            s_Pokhara = 0
            s_Chennai = 0
            s_Nepalgunj = 0
            s_Simara = 0
            s_Bhairahawa = 0
            s_Bharatpur = 0
            s_Dhangadi = 0
            s_Bhadrapur = 0
            s_Rajbiraj = 0
        
    elif (Source == 'Biratnagar'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Kathmandu = 0
            s_Banglore = 0
            s_Mumbai = 0
            s_Biratnagar = 1
            s_Janakpur = 0
            s_Pokhara = 0
            s_Chennai = 0
            s_Nepalgunj = 0
            s_Simara = 0
            s_Bhairahawa = 0
            s_Bharatpur = 0
            s_Dhangadi = 0
            s_Bhadrapur = 0
            s_Rajbiraj = 0
        
    elif (Source == 'Janakpur'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Kathmandu = 0
            s_Banglore = 0
            s_Mumbai = 0
            s_Biratnagar = 0
            s_Janakpur = 1
            s_Pokhara = 0
            s_Chennai = 0
            s_Nepalgunj = 0
            s_Simara = 0
            s_Bhairahawa = 0
            s_Bharatpur = 0
            s_Dhangadi = 0
            s_Bhadrapur = 0
            s_Rajbiraj = 0
        
    elif (Source == 'Pokhara'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Kathmandu = 0
            s_Banglore = 0
            s_Mumbai = 0
            s_Biratnagar = 0
            s_Janakpur = 0
            s_Pokhara = 1
            s_Chennai = 0
            s_Nepalgunj = 0
            s_Simara = 0
            s_Bhairahawa = 0
            s_Bharatpur = 0
            s_Dhangadi = 0
            s_Bhadrapur = 0
            s_Rajbiraj = 0
        
    elif (Source == 'Chennai'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Kathmandu = 0
            s_Banglore = 0
            s_Mumbai = 0
            s_Biratnagar = 0
            s_Janakpur = 0
            s_Pokhara = 0
            s_Chennai = 1
            s_Nepalgunj = 0
            s_Simara = 0
            s_Bhairahawa = 0
            s_Bharatpur = 0
            s_Dhangadi = 0
            s_Bhadrapur = 0
            s_Rajbiraj = 0
        
    elif (Source == 'Nepalgunj'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Kathmandu = 0
            s_Banglore = 0
            s_Mumbai = 0
            s_Biratnagar = 0
            s_Janakpur = 0
            s_Pokhara = 0
            s_Chennai = 0
            s_Nepalgunj = 1
            s_Simara = 0
            s_Bhairahawa = 0
            s_Bharatpur = 0
            s_Dhangadi = 0
            s_Bhadrapur = 0
            s_Rajbiraj = 0
        
    elif (Source == 'Simara'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Kathmandu = 0
            s_Banglore = 0
            s_Mumbai = 0
            s_Biratnagar = 0
            s_Janakpur = 0
            s_Pokhara = 0
            s_Chennai = 0
            s_Nepalgunj = 0
            s_Simara = 1
            s_Bhairahawa = 0
            s_Bharatpur = 0
            s_Dhangadi = 0
            s_Bhadrapur = 0
            s_Rajbiraj = 0
        
    elif (Source == 'Bhairahawa'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Kathmandu = 0
            s_Banglore = 0
            s_Mumbai = 0
            s_Biratnagar = 0
            s_Janakpur = 0
            s_Pokhara = 0
            s_Chennai = 0
            s_Nepalgunj = 0
            s_Simara = 0
            s_Bhairahawa = 1
            s_Bharatpur = 0
            s_Dhangadi = 0
            s_Bhadrapur = 0
            s_Rajbiraj = 0
        
    elif (Source == 'Bharatpur'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Kathmandu = 0
            s_Banglore = 0
            s_Mumbai = 0
            s_Biratnagar = 0
            s_Janakpur = 0
            s_Pokhara = 0
            s_Chennai = 0
            s_Nepalgunj = 0
            s_Simara = 0
            s_Bhairahawa = 0
            s_Bharatpur = 1
            s_Dhangadi = 0
            s_Bhadrapur = 0
            s_Rajbiraj = 0

    elif (Source == 'Dhangadi'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Kathmandu = 0
            s_Banglore = 0
            s_Mumbai = 0
            s_Biratnagar = 0
            s_Janakpur = 0
            s_Pokhara = 0
            s_Chennai = 0
            s_Nepalgunj = 0
            s_Simara = 0
            s_Bhairahawa = 0
            s_Bharatpur = 0
            s_Dhangadi = 1
            s_Bhadrapur = 0
            s_Rajbiraj = 0
        
    elif (Source == 'Bhadrapur'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Kathmandu = 0
            s_Banglore = 0
            s_Mumbai = 0
            s_Biratnagar = 0
            s_Janakpur = 0
            s_Pokhara = 0
            s_Chennai = 0
            s_Nepalgunj = 0
            s_Simara = 0
            s_Bhairahawa = 0
            s_Bharatpur = 0
            s_Dhangadi = 0
            s_Bhadrapur = 1
            s_Rajbiraj = 0
        
    elif (Source == 'Rajbiraj'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Kathmandu = 0
            s_Banglore = 0
            s_Mumbai = 0
            s_Biratnagar = 0
            s_Janakpur = 0
            s_Pokhara = 0
            s_Chennai = 0
            s_Nepalgunj = 0
            s_Simara = 0
            s_Bhairahawa = 0
            s_Bharatpur = 0
            s_Dhangadi = 0
            s_Bhadrapur = 0
            s_Rajbiraj = 1




    Destination = request.form["Destination"]
    if (Destination == 'Cochin'):
            d_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0
            d_Cochin = 1
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0
        
    elif (Destination == 'Delhi'):
            d_Delhi = 1
            d_Hyderabad = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0

        

    elif (Destination == 'Hyderabad'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 1
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0

    elif (Destination == 'Kolkata'):
            d_Delhi = 0
            d_Kolkata = 1
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0
        
    elif (Destination == 'Kathmandu'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 1
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0
        
    elif (Destination == 'Banglore'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 1
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0
        
    elif (Destination == 'Mumbai'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 1
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0
        
    elif (Destination == 'Biratnagar'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 1
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0
        
    elif (Destination == 'Janakpur'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 1
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0
        
    elif (Destination == 'Pokhara'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 1
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0
        
    elif (Destination == 'Chennai'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 1
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0
        
    elif (Destination == 'Nepalgunj'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 1
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0
        
    elif (Destination == 'Simara'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 1
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0
        
    elif (Destination == 'Bhairahawa'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 1
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0

    elif (Destination == 'Bharatpur'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 1
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0
        
    elif (Destination == 'Dhangadi'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 1
            d_Bhadrapur = 0
            d_Rajbiraj = 0
        
    elif (Destination == 'Bhadrapur'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 1
            d_Rajbiraj = 0
        
    elif (Destination == 'Rajbiraj'):
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 1

    else:
            d_Delhi = 0
            d_Kolkata = 0
            d_Cochin = 0
            d_Hyderabad = 0
            d_Kathmandu = 0
            d_Banglore = 0
            d_Mumbai = 0
            d_Biratnagar = 0
            d_Janakpur = 0
            d_Pokhara = 0
            d_Chennai = 0
            d_Nepalgunj = 0
            d_Simara = 0
            d_Bhairahawa = 0
            d_Bharatpur = 0
            d_Dhangadi = 0
            d_Bhadrapur = 0
            d_Rajbiraj = 0


    data = {
    'Total_Stops': [Total_stops],
    'Journey_day': [Journey_day],
    'Journey_month': [Journey_month],
    'Journey_year': [Journey_year],
    'Dep_hour': [Dep_hour],
    'Dep_min': [Dep_min],
    'Arrival_hour': [Arrival_hour],
    'Arrival_min': [Arrival_min],
    'Duration_hours': [dur_hour],  # Assuming the flight duration is 1 hour
    'Duration_mins': [dur_min],
    'Airline_Air India': [Air_India],
    'Airline_Buddha Air': [Buddha_Air],
    'Airline_GoAir': [GoAir],
    'Airline_IndiGo': [Jet_Airways],
    'Airline_Jet Airways': [Jet_Airways],  # Assuming the flight is not Jet Airways
    'Airline_Multiple carriers': [Multiple_carriers],
    'Airline_Shree Airlines': [Shree_Airlines],
    'Airline_SpiceJet': [SpiceJet],
    'Airline_Vistara': [Vistara],
    'Airline_Yeti Airlines': [Yeti_Airlines],
    'Source_Bhadrapur': [s_Bhadrapur],
    'Source_Bhairahawa': [s_Bhairahawa],
    'Source_Bharatpur': [s_Bharatpur],
    'Source_Biratnagar': [s_Biratnagar],
    'Source_Chennai': [s_Chennai],
    'Source_Delhi': [s_Delhi],
    'Source_Dhangadi': [s_Dhangadi],
    'Source_Janakpur': [s_Janakpur],
    'Source_Kathmandu': [s_Kathmandu],  # Kathmandu is the source
    'Source_Kolkata': [s_Kolkata],
    'Source_Mumbai': [s_Mumbai],
    'Source_Nepalgunj': [s_Nepalgunj],
    'Source_Pokhara': [s_Pokhara],
    'Source_Rajbiraj': [s_Rajbiraj],
    'Source_Simara': [s_Simara],
    'Destination_Bhadrapur': [d_Bhadrapur],
    'Destination_Bhairahawa': [d_Bhairahawa],
    'Destination_Bharatpur': [d_Bharatpur],
    'Destination_Biratnagar': [d_Biratnagar],
    'Destination_Cochin': [d_Cochin],
    'Destination_Delhi': [d_Delhi],
    'Destination_Dhangadi': [d_Dhangadi],
    'Destination_Hyderabad': [d_Hyderabad],
    'Destination_Janakpur': [d_Janakpur],
    'Destination_Kathmandu': [d_Kathmandu],  # Kathmandu is the destination
    'Destination_Kolkata': [d_Kolkata],
    'Destination_Nepalgunj': [d_Nepalgunj],  # Nepalgunj is the destination
    'Destination_New Delhi': [d_Delhi],
    'Destination_Pokhara': [d_Pokhara],
    'Destination_Rajbiraj': [d_Rajbiraj],
    'Destination_Simara': [d_Simara],
}

    features = pd.DataFrame(data)
    prediction = model.predict(features)
    output = round(prediction[0], 2)
    
    session["input_data"] = {
        "dep_time": date_dep,
        "username": username,
        "arrival_time": date_arr,
        "source": Source,
        "destination": Destination,
        "stops": Total_stops,
        "airline": airline,
        "price": output,
        "current_datetime": get_current_datetime(),
        "country": country
        }
    
    session["predicted_price"] = output
    return render_template('home.html', prediction_text="Your Flight price is ${}".format(output),username=username)

        

@app.route('/show', methods=['POST', 'GET'])
@login_required
def show():
    if current_user.is_authenticated:
        username = current_user.username
    input_data = session.get("input_data", None)

    if input_data:
        username = current_user.username if current_user.is_authenticated else "Guest"
        email = current_user.email
        contact = current_user.contact
        return render_template('show.html', dep_time=input_data["dep_time"], arrival_time=input_data["arrival_time"],
                               source=input_data["source"], destination=input_data["destination"],
                               stops=input_data["stops"], airline=input_data["airline"], price=input_data["price"],show=username,email=email, contact=contact, current_datetime=input_data["current_datetime"],country = input_data["country"],username=username)
    else:
        flash("Please provide data for prediction before showing details.", 'error')
        return redirect(url_for('show'),username=username)
    

if __name__ == "__main__":
    app.run(debug=True)
