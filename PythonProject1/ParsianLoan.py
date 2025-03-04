from sqlalchemy import (
    Column, Integer, BigInteger, String, Date, DateTime, Numeric, Float
)
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class ParsianLoan(Base):
    __tablename__ = "parsian_loan"

    id = Column("id", BigInteger, primary_key=True, autoincrement=True)
    insert_sysdate = Column("insert_sysdate", DateTime, nullable=False, default=datetime.utcnow)
    branch_code = Column("branch_code", Integer, nullable=False)
    branchname = Column("branchname", String(100), nullable=True)

    client_id = Column("client_id", Integer, nullable=True)
    loan_file_numberr = Column("loan_file_numberr", BigInteger, nullable=True)

    sit_flag = Column("sit_flag", String(1), nullable=True)
    interest_rate = Column("interest_rate", Numeric(19, 2), nullable=True)
    total_repayment_up_to_now = Column("total_repayment_up_to_now", Numeric(28, 8), nullable=True)
    commission_amount_remain = Column("commission_amount_remain", Numeric(28, 8), nullable=True)
    charge = Column("charge", Numeric(28, 8), nullable=True)
    discount = Column("discount", Numeric(28, 8), nullable=True)

    advance_pay_to_total_cash = Column("advance_pay_to_total_cash", Numeric(28, 8), nullable=True)
    advance_pay_to_remain_non_cash = Column("advance_pay_to_remain_non_cash", Numeric(28, 8), nullable=True)
    is_installment = Column("is_installment", String(1), nullable=True)
    interest_sum = Column("interest_sum", Numeric(28, 8), nullable=True)

    installment_number_remain = Column("installment_number_remain", Integer, nullable=True)
    receivable_installment_number = Column("receivable_installment_number", Integer, nullable=True)
    first_passed = Column("first_passed", Date, nullable=True)
    total_payment_up_to_now = Column("total_payment_up_to_now", Numeric(28, 8), nullable=True)
    finalized_loan_amount = Column("finalized_loan_amount", Numeric(28, 8), nullable=True)
    penalty = Column("penalty", Numeric(28, 8), nullable=True)

    first_payment_date_in_du = Column("first_payment_date_in_du", Date, nullable=True)
    principal_sum = Column("principal_sum", Numeric(28, 8), nullable=True)
    advance_pay = Column("advance_pay", Numeric(28, 8), nullable=True)

    sit_duration = Column("sit_duration", Integer, nullable=True)
    sit_duration_day = Column("sit_duration_day", Integer, nullable=True)
    sit_distribute_phases = Column("sit_distribute_phases", Integer, nullable=True)
    sit_fast_receive_percent = Column("sit_fast_receive_percent", Float, nullable=True)

    frequency = Column("frequency", Integer, nullable=True)
    customer_obligation_amount = Column("customer_obligation_amount", Numeric(28, 8), nullable=True)
    customer_share_cash_amount = Column("customer_share_cash_amount", Numeric(28, 8), nullable=True)
    customer_share_non_cash_amount = Column("customer_share_non_cash_amount", Numeric(28, 8), nullable=True)

    bank_share_cash_amount = Column("bank_share_cash_amount", Numeric(28, 8), nullable=True)
    bank_share_non_cash_amount = Column("bank_share_non_cash_amount", Numeric(28, 8), nullable=True)
    first_over_due = Column("first_over_due", Date, nullable=True)  # اصلاح شده
    loan_duration_day = Column("loan_duration_day", Integer, nullable=True)

    loan_file_number = Column("loan_file_number", BigInteger, nullable=True)
    create_date = Column("create_date", Date, nullable=True)
    long_title = Column("long_title", String(255), nullable=True)
    status = Column("status", String(255), nullable=True)
    contract = Column("contract", String(255), nullable=True)

    approval_amount = Column("approval_amount", Numeric(28, 8), nullable=True)
    title = Column("title", String(255), nullable=True)
    inc_commission_amount = Column("inc_commission_amount", Numeric(28, 8), nullable=True)
    interest_amount = Column("interest_amount", Numeric(28, 8), nullable=True)

    obligation_penalty = Column("obligation_penalty", Numeric(28, 8), nullable=True)
    passed_date = Column("passed_date", Date, nullable=True)
    penalty_interest = Column("penalty_interest", Numeric(28, 8), nullable=True)

    to_due_date = Column("to_due_date", Numeric(28, 8), nullable=True)
    to_end_of_month = Column("to_end_of_month", Numeric(28, 8), nullable=True)
    due_date = Column("due_date", Date, nullable=True)

    def __repr__(self):
        return f"<ParsianLoan(id={self.id}, branch_code={self.branch_code}, client_id={self.client_id})>"
