from StringIO import StringIO
import json
import os
import pycurl
import time
import pprint

class ReceptivitiAPI:

    def __init__(self):
        with open('creds.json', 'r') as creds:
            receptiviti_creds = json.load(creds)
        self.api_key = receptiviti_creds['receptiviti']['api_key']
        self.secret_key = receptiviti_creds['receptiviti']['secret_key']
        self.url = 'https://app.receptiviti.com/v2/api/person'

        api_key_header = 'X-API-KEY: ' + self.api_key
        secret_key_header = 'X-API-SECRET-KEY: ' + self.secret_key

        self.headers = [api_key_header,
                secret_key_header,
                'Accept: application/json',
                'Content-Type: application/json']

    # Takes text string
    # Returns the result of posting the text string to the Receptiviti API as
    # the content of a new person
    def post_contents(self, text):
        print "Posting payload to Reciptiviti..."
        payload = {
          "name": "None",
          "gender": "0",
          "content": {
              "content_source": "0",
              "language_content": text
          },
          "person_handle": time.time()
        }

        post_data = json.dumps(payload)

        buffer = StringIO()
        c = pycurl.Curl()
        c.setopt(c.URL, self.url)
        c.setopt(c.HTTPHEADER, self.headers)
        c.setopt(c.POST, 1)
        c.setopt(c.POSTFIELDS, post_data)
        c.setopt(c.WRITEDATA, buffer)

        c.perform()
        body = json.loads(buffer.getvalue())
        return body
