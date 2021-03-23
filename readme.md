## IoT Device Identification Challenge
My solution for Device Identification Kaggle challenge - https://www.kaggle.com/c/cybersecprague2019-challenge/overview

- Example train data in example-train.json
- Data description in data-readme.txt

For most columns, I extracted array of strings and converted them via MultiLabelBinarizer. I also filtered these labels by frequency, e.g. at least three times in train dataset. I also unified some labels like similar models of the same television and, for example, added printer label whenever there was a substring "printer" (see functions extract_upnp_labels and extract_ssdp_labels). This is how `known_*` arrays were created.

Final accuracy was 0.95287.

#### Other people solutions
- https://github.com/kartol/Device-Identification-challenge/blob/master/cybersecprague2019.ipynb
- https://github.com/xct/kaggle_device_identification_challenge/blob/master/kaggle_device_identification.ipynb

#### Mac address database
 - https://macaddress.io/database/macaddress.io-db.json
