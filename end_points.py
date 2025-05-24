import os
from models import engine, Session, User, bcrypt, Base, Data
from pipeline import extract_text_from_doc, generate_gap_summary, mock_interview
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
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
def home():
    return render_template("home_page.html")

@app.route('/data_collector', methods=['POST', 'GET'])
@login_required
def data():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST', 'GET'])
@login_required
def chatbot():
    return render_template('mock_interview.html')


@app.route('/process', methods=['POST'])
@login_required
def process():
    resume_file = request.files['resume']
    job_description = request.form['job_description']
    uploaded_folder = "data"
    os.makedirs(uploaded_folder, exist_ok=True)

    if resume_file:
        filename = resume_file.filename
        filepath = os.path.join(uploaded_folder, filename)
        resume_file.save(filepath)

        parsed_data = extract_text_from_doc(filepath)
        resume_text = parsed_data[0].page_content
        summary = generate_gap_summary(resume_text, job_description)

        session = Session()
        new_data = Data(
            summary=summary,
            resume_data=resume_text,
            job_description=job_description,
            user_id=current_user.id
        )
        session.add(new_data)
        session.commit()

        return render_template("response.html", data=summary)

    return "Error: No file uploaded", 404


@app.route('/mock_interview', methods=['POST', 'GET'])
@login_required
def interview():

    try:
        data = request.get_json(force=True)
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"response": "Query is missing"}), 400

        response = mock_interview(query)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 400



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
            return redirect(url_for('data'))
        else:
            flash('Invalid credentials.', 'danger')
            return redirect(url_for('signin'))

    return render_template('sign_in.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        Base.metadata.create_all(engine)
    app.run(debug=True)

