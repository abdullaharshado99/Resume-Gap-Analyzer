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
def home():
    return render_template("home_page.html")


@app.route('/data_collector', methods=['POST', 'GET'])
@login_required
def data():
    session = get_session()
    user_data = session.query(Data).filter_by(user_id=current_user.id).all()
    return render_template('process.html', user_data=user_data)


@app.route('/get_resume/<int:id>')
@login_required
def get_resume(id):
    session = get_session()
    resume = session.query(Data).filter_by(id=id, user_id=current_user.id).first()

    if resume:
        return jsonify({
            'job_description': resume.job_description
        })
    else:
        return jsonify({'error': 'Resume not found'}), 404


@app.route('/delete_resume/<int:resume_id>', methods=['DELETE'])
@login_required
def delete_resume(resume_id):
    session = get_session()
    try:
        resume = session.query(Data).filter_by(id=resume_id, user_id=current_user.id).first()
        if not resume:
            return jsonify({'error': 'Resume not found'}), 404

        session.delete(resume)
        session.commit()

        return jsonify({'success': True, 'message': 'Resume deleted successfully'})

    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()



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
            filename=filename,
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
        raw_data = request.get_json(force=True)
        query = raw_data.get("query", "").strip()
        difficulty = raw_data.get("difficulty", "medium").strip().lower()

        if not query:
            return jsonify({"response": "Query is missing"}), 400

        response = mock_interview(query, difficulty)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 400



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        session = get_session()
        existing_user = session.query(User).filter_by(email=email).first()

        if existing_user:
            flash('Email already registered.', 'danger')
            return redirect(url_for('signup'))

        new_user = User(firstname=firstname, lastname=lastname, username=username, email=email)
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

