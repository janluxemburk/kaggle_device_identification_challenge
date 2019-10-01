## IoT Device Identification Challenge
My solution for kaggle challenge - https://www.kaggle.com/c/cybersecprague2019-challenge/overview. This is my first machine-learning project.

- Example train data in example-train.json
- Data description in data-readme.txt

For most columns, I extracted array of strings and converted them via MultiLabelBinarizer. I also filtered these labels by frequency, e.g. atleast three times in train dataset. I also unified some labels like similar models of the same television or, for example, I added printer label whenever there was a substring "printer" (see functions extract_upnp_labels and extract_ssdp_labels). This is how `known_*` arrays were created.

Final accuracy was 0.95287.

### Mac address database
 - https://macaddress.io/database/macaddress.io-db.json