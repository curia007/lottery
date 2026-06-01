from apscheduler.schedulers.blocking import BlockingScheduler
from multiprocessing import Process
import os

def run_pick3():
    os.system("python scrape_idaho_pick3.py")

def run_cash():
    os.system("python scrape_idaho_cash.py")

def run_lotto():
    os.system("python scrape_lotto_america.py")

def run_all():
    p1 = Process(target=run_pick3)
    p2 = Process(target=run_cash)
    p3 = Process(target=run_lotto)

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()

scheduler = BlockingScheduler()

# Run all scripts every day at 7 PM
scheduler.add_job(run_all, 'cron', hour=19, minute=0)

print("Scheduler running...")
scheduler.start()
