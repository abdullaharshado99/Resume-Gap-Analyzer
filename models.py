import os
from flask_bcrypt import Bcrypt
from flask_login import UserMixin
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, JSON

bcrypt = Bcrypt()

os.makedirs("instance", exist_ok=True)

DATABASE_URL = 'sqlite:///instance/users.db'

engine = create_engine(DATABASE_URL, echo=True)
Session = sessionmaker(bind=engine)

class Base(DeclarativeBase):
    pass

class User(Base, UserMixin):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    firstname = Column(String, nullable=False)
    lastname = Column(String, nullable=False)
    username = Column(String(150), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    password = Column(String(256), nullable=False)

    data = relationship('Data', back_populates='user', cascade='all, delete-orphan')

    def set_password(self, raw_password):
        self.password = bcrypt.generate_password_hash(raw_password).decode('utf-8')

    def check_password(self, raw_password):
        return bcrypt.check_password_hash(self.password, raw_password)

    def __repr__(self):
        return f"<User {self.username}, {self.email}>"

class Data(Base):
    __tablename__ = 'users_data'

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    summary = Column(String(5000), nullable=False)
    resume_data = Column(String(5000), nullable=False)
    job_description = Column(String(5000), nullable=False)
    resume_match_score = Column(JSON, nullable=True)
    interview_match_score = Column(JSON, nullable=True)


    user = relationship('User', back_populates='data')
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    def __repr__(self):
        return f"<Data for User ID {self.user}>"

if __name__ == '__main__':
    Base.metadata.create_all(engine)

    session = Session()
    session.query(Data).delete()
    session.query(User).delete()
    session.commit()