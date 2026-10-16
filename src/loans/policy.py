"""The bank's process rules, in one place.

These are frame rules enforced in code: the agent cannot change them, and the
evals check them. Each value notes where it comes from.
"""

# A customer who does not answer is cancelled only after this many days
# without an answer, counted from the agent's last follow-up message, never
# earlier. From BPI Challenge 2017: the bank cancelled applications
# automatically about 26 days after the last follow-up call.
NO_ANSWER_CANCEL_DAYS = 26

# The first request for missing information must be sent within this many
# days of receiving the application. Regulation B (12 CFR 1002.9) requires a
# lender to notify the applicant within 30 days of receiving an incomplete
# application.
INCOMPLETE_NOTICE_DAYS = 30
