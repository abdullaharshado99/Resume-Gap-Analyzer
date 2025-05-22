import os
from models import engine, Session, User, bcrypt, Base
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

bcrypt.init_app(app)

def get_session():
    return Session(bind=engine)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'signin'

@login_manager.user_loader
def load_user(user_id):
    session = get_session()
    return session.get(User, int(user_id))

@app.route('/')
@login_required
def index():
    return render_template("index.html", user=current_user)

@app.route('/process', methods=['POST'])
def process():
    resume_file = request.files['resume']
    job_description = request.form['job_description']
    uploaded_folder = "data"
    os.makedirs(uploaded_folder, exist_ok=True)

    if resume_file:
        filename = resume_file.filename
        filepath = os.path.join(uploaded_folder, filename)
        resume_file.save(filepath)
        return f"Received file: {filename}<br>Job Description: {job_description[:200]}..."

    return "Error: No file uploaded"

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        session = get_session()
        existing_user = session.query(User).filter_by(email=email).first()

        if existing_user:
            flash('Email already registered.', 'danger')
            return redirect(url_for('signup'))

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        session.add(new_user)
        session.commit()
        flash('Account created. Please log in.', 'success')
        return redirect(url_for('signin'))

    return render_template('sign_up.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        session = get_session()
        user = session.query(User).filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials.', 'danger')
            return redirect(url_for('signin'))

    return render_template('sign_in.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('signin'))

if __name__ == '__main__':
    with app.app_context():
        Base.metadata.create_all(engine)
    app.run(debug=True)

