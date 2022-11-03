# v0.10
# Dependencies: OpenSSL installed (to handle https/SSL), requests module
# Notes: The requests module handles most of the heavy lifting
#       This will loop through all results in a query, 50 at a time,
#       saving individual JSON files to the same directory as the script


#You will need to install the 'lxml' parser via the python terminal (not this script) using: python -m pip install lxml
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime  # used to name json files
from time import sleep
import json
import sys
import pandas as pd
import os

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\DownloadingArticles')


d = datetime.today()
today = d.strftime('%Y-%m-%d')

#Developers Website: https://www.lexisnexis.com/lextalk/developers/default.aspx
#Test Application:  https://solutions-bis.lexisnexis.com/wsapi/sign-in/
#Building Search String: https://www.lexisnexis.com/help/CU/Building_a_Search/Searches.htm
#Search Commands: https://help.lexisnexis.com/tabula-rasa/rosetta/hlead_ref-reference?lbu=GB&locale=en_GB&audience=business

############ Build your search ###########

client_id = 'BPKVKVNHFTFRPDRWFFRSVXV2QXNPDH' #Enter cliet ID within the quotes.
secret = 'GPWPNBNFVTTQFZQPNGV1CFGMNMRXKSHXZNZSCTRH'  #Enter secret within the quotes.

#Search String: 
query = 'covert OR clandestine' #Enter the terms you would like to search for (eg. hlead('air base' OR 'air strike'))

#Search Range:
#You can leave these blank, but if you enter the end date, you must also enter the start date.
#Note that this range is inclusive (i.e. it will include all stories from the start day and the end day).
sdate = '1979-01-01' #Enter the start date for your search (yyyy-mm-dd)
edate = '1980-12-31' #Enter the end date for your search (yyyy-mm-dd)

#Sources
sources = [''] #Enter each source as a new list element (e.g. ['MTA2OTUwNQ','MTA2OTIwMQ', 'MTA1MjQ3Mw']) Source IDs are included in the 'LexisNexisSources' document.

#'MTA2MTk3Mg','MTA1MjQ3Mw','MTA2MzA3Ng','MTA2MzkzNg','MTA1MzA5OQ','MTA2NDYxMA','MTA2NDYyNA','MTA1MzI3Mw','MTA1NDUyOA','MTA2NzU3NQ','MTA2OTUwNQ','MTA2NzgzOQ','MTA2OTE3MQ','MTA2OTEwMQ'

#Value for Grouping Duplicates
duplicates = 'ModerateSimilarity' #HighSimilarity or ModerateSimilarity or leave blank for no grouping.

#Values for Language
#Languages is left blank by default (in which case all language are searched), currently only one language can be selected at a time.
languages = ['English'] #Options for Language: English, French, German, Spanish, Dutch, Portuguese, Italian, Russian, Japanese, Danish, Swedish, Norwegian, Indonesian, Vietnamese, Romanian, Turkish, Korean, Greek, Arabic, Afrikans, Croatian, Czech, Catalan, Slovenian, Estonian, Chinese, Malay, Hungarian, Lithuanian, Bulgarian, Finnish, Polish, Slovak

#Search Type
searchtype = 'Boolean' #DynamicAnd is used by default (leave blank) or choose one of these options: DynamicOr, NaturalLanguageOr, NaturalLanguageAnd, Boolean
 
#Adjusts the number of results to return per outout document
top = 50



#NO USER INPUT IS NEEDED BEYOND THIS POINT


############# Building the Filter ###############
#Dates
if(sdate == ''):
    sdate = ''
else:
    sdate = "Date ge " + sdate + "T00:00:00Z "
    
if(edate == ''):
    edate = ''
else:
    edate = "and Date le " + edate + "T23:59:59Z "

#Sources
if(sources == ['']):
    sourcefilter = ''
elif(len(sources) == 1):
    sourcefilter = "and Source/Id eq '" + sources[0] + "'"
elif(len(sources) > 1):
    sourcefilter = ["and (Source/Id eq '" + sources[0] + "'"]
    for i in range(1,len(sources)):
        sourcefilter.append(" or Source/Id eq " + "'" + sources[i] + "'")
    sourcefilter = ''.join(sourcefilter) + ") "
    
#Search Type
if(searchtype == ''):
    searchtype = ''
elif(searchtype == 'DynamicAnd'):
    searchtype = "and SearchType eq LexisNexis.ServicesApi.SearchType'DynamicAnd' "
elif(searchtype == 'DynamicOr'):
    searchtype = "and SearchType eq LexisNexis.ServicesApi.SearchType'DynamicOr' "
elif(searchtype == 'NaturalLanguageOr'):
    searchtype ="and SearchType eq LexisNexis.ServicesApi.SearchType'NaturalLanguageOr' "
elif(searchtype == 'NaturalLanguageAnd'):
    searchtype ="and SearchType eq LexisNexis.ServicesApi.SearchType'NaturalLanguageAnd' "
elif(searchtype == 'Boolean'):
    searchtype ="and SearchType eq LexisNexis.ServicesApi.SearchType'Boolean' "
else:
    print('You did not enter a valid parameter for option: searchtype. Please try again.')
    sys.exit()

#Duplicate Grouping
if(duplicates == 'ModerateSimilarity'):
    duplicates = "and GroupDuplicates eq LexisNexis.ServicesApi.GroupDuplicates'ModerateSimilarity'" 
elif(duplicates == 'HighSimilarity'):
    duplicates = "and GroupDuplicates eq LexisNexis.ServicesApi.GroupDuplicates'HighSimilarity'" 
elif(duplicates == ''):
    duplicates ==''
else:
    print('You did not enter a valid parameter for option: duplicates. Please try again.')
    sys.exit()
#Language Filtering
if(languages == ['']):
    languagefilter = ''
elif(len(languages) == 1):
    languagefilter = "and Language eq LexisNexis.ServicesApi.Language" + "'" + languages[0] + "' "
elif(len(languages) > 1):
    languagefilter = ["and (Language eq LexisNexis.ServicesApi.Language" + "'" + languages[0] + "' "]
    for i in range(1,len(sources)):
        languagefilter.append(" or Language eq LexisNexis.ServicesApi.Language" + "'" + languages[i] + "'")
    languagefilter = ''.join(languagefilter) + ") "

#Filter
filter = (sdate + edate + 
         sourcefilter +
         languagefilter +
         searchtype +
         duplicates)
if(filter == ''):
    filter = None

############# Begin Function Definitions #############

def get_token(client_id, secret):
    """Gets Authorizaton token to use in other requests."""
    auth_url = 'https://auth-api.lexisnexis.com/oauth/v2/token'
    payload = ('grant_type=client_credentials&scope=http%3a%2f%2f'
                'oauth.lexisnexis.com%2fall')
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    r = requests.post(
            auth_url,
            auth=HTTPBasicAuth(client_id, secret),
            headers=headers,
            data=payload)
    json_data = r.json()
    return json_data['access_token']

def build_url(content='News', query='', skip=0, expand='Document', top=50, filter=None):
    """Builds the URL part of the request to Web Services API."""
    if filter != None:  # Filter is an optional parameter
        api_url = ('https://services-api.lexisnexis.com/v1/' + content +
                    '?$expand=' + expand + '&$search=' + query +
                    '&$skip=' + str(skip) + '&$top=' + str(top) +
                    '&$filter=' + filter)
    else:
        api_url = ('https://services-api.lexisnexis.com/v1/' + content +
                    '?$expand=' + expand + '&$search=' + query +
                    '&$skip=' + str(skip) + '&$top=' + str(top))
    return api_url


def build_header(token):
    """Builds the headers part of the request to Web Services API."""
    headers = {'Accept': 'application/json;odata.metadata=minimal',
                'Connection': 'Keep-Alive',
                'Host': 'services-api.lexisnexis.com'}
    headers['Authorization'] = 'Bearer ' + token
    return headers


def get_result_count(json_data):
    """Gets the number of results from @odata.count in the response"""
    return json_data['@odata.count']


def time_now():
    """Gets current time to the second."""
    now = datetime.now()
    return now.strftime('%Y-%m-%d-%H%M%S')

############# End Function Defnitions #############

############# Begin business logic #############

token = get_token(client_id, secret)  # 1 token will work for multiple requests
request_headers = build_header(token)
#Sets starting skip
skip_value = 0 

col_names = ("Index","ResultId","Jurisdiction","Location","ContentType","Byline","WordLength","WebNewsUrl","Geography","NegativeNews","Language","Industry","People","Subject","Section","Company","PublicationType","Publisher","GroupDuplicates","InternationalLocation","SearchType","Date","Keyword","Title","DocumentContent@odata.mediaReadLink","DocumentContent@odata.mediaContentType","Overview","Extracts","IsCitationMatch","Document.DocumentId","Document.DocumentIdType","Document.Content","Document.Citation","Source.Id","Source.Name","Source.ContentType","Source.Jurisdiction","Source.Publisher","Source.AlphaCategory")

temp = pd.DataFrame(columns=col_names)
while True:
    request_url = build_url(content='News', query=query, skip=skip_value, expand='Document', top=top, filter=filter)  # Filter is set to filter=None here. Change to filter=filter to use the filter specified above
    r = requests.get(request_url, headers=request_headers)
    
    text = r.text
    res = json.loads(text) 
    
    temp = pd.concat([temp,pd.json_normalize(pd.DataFrame.from_dict(res)['value'])],ignore_index = True)

    skip_value = (skip_value + top)
    json_data = r.json()
    if skip_value > get_result_count(json_data):  # Check to see if all the results have been looped through
        break
    print("Remaining requests: " + list(r.headers.values())[13] + ' (minute/hour/day)')

    sleep(21)  # Limit 5 requests per minute (every 12 seconds)

temp['Link'] = ""
for i in range(len(temp['ResultId'])):
    temp['Link'][i] = "https://advance.lexis.com/api/document?collection=News&id=" + temp['ResultId'][i] + "&context=1516831"

temp.to_csv('MAC_79-80.csv')
