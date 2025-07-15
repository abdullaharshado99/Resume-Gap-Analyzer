import os
import logging
import secrets
import random, datetime
from flask_cors import CORS
from dotenv import load_dotenv
from flask_mail import Mail, Message
from flask import session as flask_session
from models import engine, Session, User, bcrypt, Base, Data, Admin, Announcement
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from pipeline import extract_text_from_doc, generate_gap_summary, generate_gap_score, mock_interview

secret_key = secrets.token_hex(16)

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key
app.config['WTF_CSRF_ENABLED'] = True

bcrypt.init_app(app)

def get_session():
    return Session(bind=engine)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'signin'
login_manager.login_view = 'admin-signin'


# Flask-Mail config
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='abdullaharshado99@gmail.com',
    MAIL_PASSWORD=os.getenv('email_pass')
)

CORS(app)

mail = Mail(app)

# Store OTPs temporarily
otp_store = {}


@login_manager.user_loader
def load_user(user_id):
    session = get_session()
    return session.get(User, int(user_id))


@app.route('/')
def home():
    session = Session()
    announcement = session.query(Announcement).order_by(Announcement.created_at.desc()).limit(3).all()
    return render_template('home_page.html', announcements=announcement)


@app.route('/terms')
def terms():
    return render_template("terms.html")


@app.route('/privacy')
def privacy():
    return render_template("privacy.html")


@app.route('/blog')
def blog():
    return render_template("blog.html")


@app.route('/faqs')
def faqs():
    return render_template("faqs.html")


@app.route('/tips')
def tips():
    return render_template("tips.html")


@app.route('/interview_guide')
def interview_guide():
    return render_template("interview_guide.html")

@app.route('/admin')
def dashboard():
    return render_template("admin_dashboard.html")

@app.route('/admin-homepage')
def admin_homepage():
    return render_template("admin_homepage.html")


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        session = Session()
        user = session.query(User).filter_by(email=email).first()
        if user:
            otp = str(random.randint(100000, 999999))
            otp_store[email] = {
                'otp': otp,
                'expires': datetime.datetime.now() + datetime.timedelta(minutes=5)
            }
            msg = Message('AI Career Assistant - Your OTP for Password Reset', sender=app.config['MAIL_USERNAME'], recipients=[email])
            msg.body = f'Your OTP is: {otp}, this OTP is only validate for 5 minutes, regenerate new otp after 5 minutes.'
            mail.send(msg)
            flask_session['reset_email'] = email
            flash("OTP sent to your email.", "info")
            return redirect(url_for('verify_otp'))
        else:
            flash("Email not found.", "danger")
    return render_template('forgot_password.html')


@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    email = flask_session.get('reset_email')
    if not email:
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        entered_otp = request.form['otp']
        new_password = request.form['password']
        record = otp_store.get(email)

        if not record:
            flash("OTP not found or expired. Try again.", "danger")
            return redirect(url_for('forgot_password'))

        if record['otp'] == entered_otp and datetime.datetime.now() < record['expires']:
            db = Session()
            user = db.query(User).filter_by(email=email).first()
            if user:
                user.set_password(new_password)
                db.commit()
                otp_store.pop(email, None)
                flask_session.pop('reset_email', None)
                flash("Password updated successfully!", "success")
                return redirect(url_for('signin'))
            else:
                flash("User not found.", "danger")
        else:
            flash("Invalid or expired OTP.", "danger")

    return render_template('verify_otp.html')


@app.route('/data_collector', methods=['POST', 'GET'])
@login_required
def data():
    session = get_session()
    user_data = session.query(Data).filter_by(user_id=current_user.id).all()
    return render_template('process.html', user_data=user_data, user=current_user)


@app.route('/get_resume/<int:resume_id>')
@login_required
def get_resume(resume_id):
    session = get_session()
    resume = session.query(Data).filter_by(id=resume_id, user_id=current_user.id).first()

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


@app.route('/result/<int:data_id>')
@login_required
def result(data_id):
    session = Session()
    record = session.query(Data).filter_by(id=data_id, user_id=current_user.id).first()

    if record:
        gap_score = generate_gap_score(record.summary)
        record.resume_match_score = gap_score
        session.commit()
        return render_template("response.html", data=record.summary, gap_data=gap_score)

    return "Data not found", 404


@app.route('/chatbot', methods=['POST', 'GET'])
@login_required
def chatbot():
    return render_template('mock_interview.html')


@app.route('/final_report', methods=['POST', 'GET'])
@login_required
def final_report():
    session = Session()
    record = session.query(Data).filter_by(user_id=current_user.id).order_by(Data.id.desc()).first()

    if record and record.resume_match_score:
        resume_gap_score = record.resume_match_score
        interview_dict = {"Confidence": 30, "Communication": 25, "Problem Solving": 25, "Technical Depth": 20}
    else:
        resume_gap_score = interview_dict = {}

    return render_template('final_report.html', resume_score_data=resume_gap_score, interview_score_data=interview_dict, user=current_user)


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

        return redirect(url_for('result', data_id=new_data.id))

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
        firstname = request.form.get('firstname', '').strip()
        lastname = request.form.get('lastname', '').strip()
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()

        if not all([firstname, lastname, username, email, password]):
            flash('All fields are required.', 'danger')
            return redirect(url_for('signup'))

        session = get_session()
        try:
            existing_user = session.query(User).filter(
                (User.email == email) | (User.username == username)
            ).first()

            if existing_user:
                if existing_user.email == email:
                    flash('Email already registered.', 'danger')
                else:
                    flash('Username already taken.', 'danger')
                return redirect(url_for('signup'))

            new_user = User(
                firstname=firstname,
                lastname=lastname,
                username=username,
                email=email
            )
            new_user.set_password(password)
            session.add(new_user)
            session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('signin'))
        except Exception as e:
            logging.error(e)
            session.rollback()
            flash('An error occurred during registration. Please try again.', 'danger')
            return redirect(url_for('signup'))
        finally:
            session.close()

    return render_template('sign_up.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()

        if not email or not password:
            flash('Both email and password are required.', 'danger')
            return redirect(url_for('signin'))

        session = get_session()
        try:
            user = session.query(User).filter_by(email=email).first()

            if user and user.check_password(password):
                login_user(user)
                next_page = request.args.get('next')
                flash('Logged in successfully!', 'success')
                return redirect(next_page or url_for('data'))
            else:
                flash('Invalid email or password.', 'danger')
                return redirect(url_for('signin'))
        except Exception as e:
            logging.error(e)
            flash('An error occurred during login. Please try again.', 'danger')
            return redirect(url_for('signin'))
        finally:
            session.close()

    return render_template('sign_in.html')

@app.route('/admin-signup', methods=['GET', 'POST'])
def admin_signup():
    if request.method == 'POST':
        firstname = request.form.get('firstname', '').strip()
        lastname = request.form.get('lastname', '').strip()
        admin = request.form.get('admin', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()

        if not all([firstname, lastname, admin, email, password]):
            flash('All fields are required.', 'danger')
            return redirect(url_for('admin_signup'))

        session = get_session()
        try:
            existing_admin = session.query(Admin).filter(
                (Admin.email == email) | (Admin.admin == admin)
            ).first()

            if existing_admin:
                if existing_admin.email == email:
                    flash('Email already registered.', 'danger')
                else:
                    flash('Admin already registered.', 'danger')
                return redirect(url_for('admin_signup'))

            new_admin = Admin(
                firstname=firstname,
                lastname=lastname,
                admin=admin,
                email=email
            )
            new_admin.set_password(password)
            session.add(new_admin)
            session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('admin_signin'))
        except Exception as e:
            logging.error(e)
            session.rollback()
            flash('An error occurred during registration. Please try again.', 'danger')
            return redirect(url_for('admin_signup'))
        finally:
            session.close()

    return render_template('admin_signup.html')

@app.route('/admin-signin', methods=['GET', 'POST'])
def admin_signin():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()

        if not email or not password:
            flash('Both email and password are required.', 'danger')
            return redirect(url_for('admin_signin'))

        session = get_session()
        try:
            admin = session.query(Admin).filter_by(email=email).first()

            if admin and admin.check_password(password):
                login_user(admin)
                next_page = request.args.get('next')
                flash('Logged in successfully!', 'success')
                return redirect(next_page or url_for('admin_dashboard'))
            else:
                flash('Invalid email or password.', 'danger')
                return redirect(url_for('admin_signin'))
        except Exception as e:
            logging.error(e)
            flash('An error occurred during login. Please try again.', 'danger')
            return redirect(url_for('admin_signin'))
        finally:
            session.close()

    return render_template('admin_signin.html')

@app.route('/admin-dashboard', methods=['GET', 'POST'])
@login_required
def admin_dashboard():

    session = get_session()
    if request.method == 'POST':
        message = request.form.get('message')
        if message:
            ann = Announcement(message=message)
            session.add(ann)
            session.commit()
            flash("Announcement posted.", "success")
        return redirect(url_for('admin_dashboard'))

    all_ann = session.query(Announcement).order_by(Announcement.created_at.desc()).all()
    return render_template('admin_dashboard.html', announcements=all_ann)



@app.route('/delete-announcement/<int:ann_id>', methods=['POST'])
@login_required
def delete_announcement(ann_id):
    session = get_session()
    try:
        announcement = session.query(Announcement).get(ann_id)
        if announcement:
            session.delete(announcement)
            session.commit()
            flash('Announcement deleted.', 'success')
        else:
            flash('Announcement not found.', 'danger')
    except Exception as e:
        logging.error(e)
        flash('Error deleting announcement.', 'danger')
        session.rollback()
    finally:
        session.close()
    return redirect(url_for('admin_dashboard'))  # or your announcements view



@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/admin-logout')
@login_required
def admin_logout():
    logout_user()
    return redirect(url_for('admin_homepage'))

if __name__ == '__main__':
    with app.app_context():
        Base.metadata.create_all(engine)
    app.run(debug=True)

