f_nm="20180402-114759.zip"
echo "Downloading "$f_nm
f_id="1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(
	wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$f_id -O- |\
	sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$f_id" -O $f_nm && rm -rf /tmp/cookies.txt \
&& unzip $f_nm \
&& rm $f_nm
