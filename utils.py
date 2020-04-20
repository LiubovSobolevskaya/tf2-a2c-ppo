from datetime import datetime


def get_haram_str(**kwargs):
    now = datetime.now()
    date_time = now.strftime("%H:%M:%S")

    kwargs[date_time] = date_time
    hparam_str = ','.join(
        ['%s=%s' % (k, str(kwargs[k])) for k in sorted(kwargs.keys())])

    return hparam_str
