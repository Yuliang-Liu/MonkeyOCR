
def join_path(*args):
    return '/'.join(str(s).rstrip('/') for s in args)


def get_top_percent_list(num_list, percent):
    if len(num_list) == 0:
        top_percent_list = []
    else:

        sorted_imgs_len_list = sorted(num_list, reverse=True)

        top_percent_index = int(len(sorted_imgs_len_list) * percent)

        top_percent_list = sorted_imgs_len_list[:top_percent_index]
    return top_percent_list


def mymax(alist: list):
    if len(alist) == 0:
        return 0
    else:
        return max(alist)


def parse_bucket_key(s3_full_path: str):
    s3_full_path = s3_full_path.strip()
    if s3_full_path.startswith("s3://"):
        s3_full_path = s3_full_path[5:]
    if s3_full_path.startswith("/"):
        s3_full_path = s3_full_path[1:]
    bucket, key = s3_full_path.split("/", 1)
    return bucket, key
