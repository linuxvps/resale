from sqlalchemy import Column, Integer, String, Text, SmallInteger
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class LoanFeatures(Base):
    __tablename__ = "loan_features"

    feature_id = Column(Integer, primary_key=True, autoincrement=True)
    column_name = Column(String(255), nullable=False)
    table_name = Column(String(255), nullable=False)
    importance_level = Column(SmallInteger, nullable=False)
    importance_description = Column(Text, nullable=False)

    def __repr__(self):
        return f"<LoanFeatures(feature_id={self.feature_id}, column_name='{self.column_name}', table_name='{self.table_name}', importance_level={self.importance_level})>"
