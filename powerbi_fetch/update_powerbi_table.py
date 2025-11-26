import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
import pandas as pd


ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6InJ0c0ZULWItN0x1WTdEVlllU05LY0lKN1ZuYyIsImtpZCI6InJ0c0ZULWItN0x1WTdEVlllU05LY0lKN1ZuYyJ9.eyJhdWQiOiJodHRwczovL2FuYWx5c2lzLndpbmRvd3MubmV0L3Bvd2VyYmkvYXBpIiwiaXNzIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvMzUxMDI1MDItMzliYS00NGFjLWFjYzQtYzE4NWUwYmY5Mjg1LyIsImlhdCI6MTc2Mzk5ODM1MCwibmJmIjoxNzYzOTk4MzUwLCJleHAiOjE3NjQwMDM1MDksImFjY3QiOjAsImFjciI6IjEiLCJhaW8iOiJBVVFBdS84YUFBQUFqL1JPaXdIUnJ1cEdGbk1nendheENrRm4xUzdkNFEwQVAvRmJNY3h6dHBJckFFOVQ0dU53ZXBScTVFa3ZZaDlOWDZRb3VxVFByaUxla0grWWFyU0FTUT09IiwiYW1yIjpbInB3ZCIsInJzYSJdLCJhcHBpZCI6IjA0YjA3Nzk1LThkZGItNDYxYS1iYmVlLTAyZjllMWJmN2I0NiIsImFwcGlkYWNyIjoiMCIsImRldmljZWlkIjoiZGJlZjYwNWEtZjNjYS00ODY4LWI2YzItNDZhNWZmZDNkZDNmIiwiZmFtaWx5X25hbWUiOiJBYnUgU2hrYXJhIiwiZ2l2ZW5fbmFtZSI6IlNlZ2FsIiwiaWR0eXAiOiJ1c2VyIiwiaXBhZGRyIjoiMTkyLjExNC4xMDUuMjQ2IiwibmFtZSI6IlNlZ2FsIGFidSBTaGthcmEgfCBPcmJpYSAoTmV0YWZpbSkiLCJvaWQiOiI0MjdlNTcwYi00NzFjLTQxMjgtOTM1NC1lYWI2MmQ0ZTM2NDkiLCJvbnByZW1fc2lkIjoiUy0xLTUtMjEtMTE3Mjc2NzAwMi01NzAxNzQ5ODMtMTg1MjkwMzcyOC0xMzAyMjQiLCJwdWlkIjoiMTAwMzIwMDQ1MTQwMEY1RSIsInB3ZF91cmwiOiJodHRwczovL2dvLm1pY3Jvc29mdC5jb20vZndsaW5rLz9saW5raWQ9MjIyNDE5OCIsInJoIjoiMS5BUXNBQWlVUU5ibzVyRVNzeE1HRjRMLVNoUWtBQUFBQUFBQUF3QUFBQUFBQUFBQ0VBUHNMQUEuIiwic2NwIjoidXNlcl9pbXBlcnNvbmF0aW9uIiwic2lkIjoiMDA3YmEwODktZDQ0Yi03ZDA1LTg3YTUtYmU3NDczZTRiMGIxIiwic2lnbmluX3N0YXRlIjpbImR2Y19tbmdkIiwiZHZjX2RtamQiXSwic3ViIjoiX0hRQnlhMWNUUWdlSnBwYnRyc054eHVNYWpFSHNYaTZQVUlDTmgwZUpJWSIsInRpZCI6IjM1MTAyNTAyLTM5YmEtNDRhYy1hY2M0LWMxODVlMGJmOTI4NSIsInVuaXF1ZV9uYW1lIjoic2VnYWwuYWJ1LnNoa2FyYUBuZXRhZmltLmNvbSIsInVwbiI6InNlZ2FsLmFidS5zaGthcmFAbmV0YWZpbS5jb20iLCJ1dGkiOiJjUUE1MkZpQXlrS2RpVFZ0SG9Ba0FBIiwidmVyIjoiMS4wIiwid2lkcyI6WyJiNzlmYmY0ZC0zZWY5LTQ2ODktODE0My03NmIxOTRlODU1MDkiXSwieG1zX2FjdF9mY3QiOiIzIDUiLCJ4bXNfY2MiOlsiQ1AxIl0sInhtc19mdGQiOiJBUmMtVm5lbnMxaVE4OGdkTmZJeTJTRDRyeFgwMzBrdTVSUWFTODZBZjdRQlpuSmhibU5sWXkxa2MyMXoiLCJ4bXNfaWRyZWwiOiIxIDI0IiwieG1zX3N1Yl9mY3QiOiIxMCAzIn0.nTF3V_CFb_aTk6zUVMRKi0eeUwDdeyvG0kmPguGtPnaBLmtahxVP0uOwasfJ8wHe1bKLsjuqnXBvCFg5V--BrH9rPJnJhsxzsOVQiPVoxJi5Vd9tjA4JjjJWcat8UxExrD0accmqPt2TdUYLmUf7gPaG2RWgaDsTJcNRfakz-VLKKziKdvF7EZUrN85NfRvWUg6ZiukgNQCFZCxpxV-QSERjvFDQ_GSbpO0vWVIiUlugk-eCpdXqXJE7e5sqoK1FgQHSqqr_-qGZF4EEZoQIU5ECajusjtzS2Rf2aGAiu6BodzPByrbs-YJzKicA5nn3u0FdoFMcuE4y0iAnB-MkTg.eyJhdWQiOiJodHRwczovL2FuYWx5c2lzLndpbmRvd3MubmV0L3Bvd2VyYmkvYXBpIiwiaXNzIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvMzUxMDI1MDItMzliYS00NGFjLWFjYzQtYzE4NWUwYmY5Mjg1LyIsImlhdCI6MTc2Mzk4NjEzMSwibmJmIjoxNzYzOTg2MTMxLCJleHAiOjE3NjM5OTEyNDUsImFjY3QiOjAsImFjciI6IjEiLCJhaW8iOiJBVVFBdS84YUFBQUFTRFM4ZEc5YXZsWVRHWldEY0pQV2Y4anJ6QTk2V0xpdG9vaGhncHRWUXo4YTlHQ2s4dU5mbkJieDc1bkNhbG52VkhuRHlva04xSHBOSmRMZzRhTHFlZz09IiwiYW1yIjpbInB3ZCIsInJzYSJdLCJhcHBpZCI6IjA0YjA3Nzk1LThkZGItNDYxYS1iYmVlLTAyZjllMWJmN2I0NiIsImFwcGlkYWNyIjoiMCIsImRldmljZWlkIjoiZGJlZjYwNWEtZjNjYS00ODY4LWI2YzItNDZhNWZmZDNkZDNmIiwiZmFtaWx5X25hbWUiOiJBYnUgU2hrYXJhIiwiZ2l2ZW5fbmFtZSI6IlNlZ2FsIiwiaWR0eXAiOiJ1c2VyIiwiaXBhZGRyIjoiMTkyLjExNC4xMDUuMjQ2IiwibmFtZSI6IlNlZ2FsIGFidSBTaGthcmEgfCBPcmJpYSAoTmV0YWZpbSkiLCJvaWQiOiI0MjdlNTcwYi00NzFjLTQxMjgtOTM1NC1lYWI2MmQ0ZTM2NDkiLCJvbnByZW1fc2lkIjoiUy0xLTUtMjEtMTE3Mjc2NzAwMi01NzAxNzQ5ODMtMTg1MjkwMzcyOC0xMzAyMjQiLCJwdWlkIjoiMTAwMzIwMDQ1MTQwMEY1RSIsInB3ZF91cmwiOiJodHRwczovL2dvLm1pY3Jvc29mdC5jb20vZndsaW5rLz9saW5raWQ9MjIyNDE5OCIsInJoIjoiMS5BUXNBQWlVUU5ibzVyRVNzeE1HRjRMLVNoUWtBQUFBQUFBQUF3QUFBQUFBQUFBQ0VBUHNMQUEuIiwic2NwIjoidXNlcl9pbXBlcnNvbmF0aW9uIiwic2lkIjoiMDA3YmEwODktZDQ0Yi03ZDA1LTg3YTUtYmU3NDczZTRiMGIxIiwic2lnbmluX3N0YXRlIjpbImR2Y19tbmdkIiwiZHZjX2RtamQiXSwic3ViIjoiX0hRQnlhMWNUUWdlSnBwYnRyc054eHVNYWpFSHNYaTZQVUlDTmgwZUpJWSIsInRpZCI6IjM1MTAyNTAyLTM5YmEtNDRhYy1hY2M0LWMxODVlMGJmOTI4NSIsInVuaXF1ZV9uYW1lIjoic2VnYWwuYWJ1LnNoa2FyYUBuZXRhZmltLmNvbSIsInVwbiI6InNlZ2FsLmFidS5zaGthcmFAbmV0YWZpbS5jb20iLCJ1dGkiOiJ5c1pCT0dGclFVS1A2LUdrS3lrZkFBIiwidmVyIjoiMS4wIiwid2lkcyI6WyJiNzlmYmY0ZC0zZWY5LTQ2ODktODE0My03NmIxOTRlODU1MDkiXSwieG1zX2FjdF9mY3QiOiIzIDUiLCJ4bXNfY2MiOlsiQ1AxIl0sInhtc19mdGQiOiJPYi0ydl9DWUtpMjhmdUh3eU9pVjBZRVhtRUpIYnh6RFUxTWZRdk5VSDRvQlpYVnliM0JsYm05eWRHZ3RaSE50Y3ciLCJ4bXNfaWRyZWwiOiIyMCAxIiwieG1zX3N1Yl9mY3QiOiIyIDMifQ.Yyf2HygooGE-CPl1hiPgWSQVvKT7MBs5Vfiv4JqJKqQfv2exp-BxHmLeZsshC1DltjZr6TECRwqMm7xJiY4lxKxPSAWDGCaScRFX3vpNKQS0QSudlyP1GYBWmW1vGT_NIsySVheg0Z_7g35xGd3y7cUlml4b_wLVrkvVBbsHfmQapWMlfTPoD6mrYsQAgJgGODgeakvHDYQVABx4lOKkWi3xSK4lcjX2lyPbYPTTw59OCm4l8Fhi3xVpk2FxgIp_A-0AeTeSTsQaSxG4GnYr1fE3JXFBuDDpurlqDwTzOOiGD3Rah__fcnEeY2emCubeIiauIpCiAcKNkD_lRNOefg"

DATASET_ID = "815d1efb-8797-44dc-92f1-9e6af78cb505"

DAX_QUERY = """
EVALUATE
CALCULATETABLE(
    ADDCOLUMNS(
        SUMMARIZECOLUMNS(
            'devices_status'[DeviceId],
            'devices_status'[Day],
            'CS SQL query'[Is FloLive],
            'enterprises'[enterprises.distributorName],
            'enterprises'[enterprises.country],
            'CS SQL query'[FarmName],
            'CS SQL query'[created_at],
            'CS SQL query'[Last_transmission],
            'devices_status'[SwVersion]
        ),
        "AvgBatteryVoltage", [AvgBatteryVoltage],
        "AvgRSSI", [AvgRSSI],
        "BatterySlope3d", [BatterySlope3d],
        "BatterySlope7d", [BatterySlope7d],
        "HeartbeatCount", [HeartbeatCount],
        "HeartbeatSlope3d", [HeartbeatSlope3d],
        "HeartbeatSlope7d", [HeartbeatSlope7d],
        "RSSISlope3d", [RSSISlope3d],
        "RSSISlope7d", [RSSISlope7d],
        "MaxGapMinutes", [MaxGapMinutes]
    ),
    KEEPFILTERS( 'CS SQL query'[Device_Type] IN { "One" } ),
    KEEPFILTERS( 'CS SQL query'[ConnectionStatusUpdated] IN { "Connected", "Not Connected" } ),
    KEEPFILTERS( NOT ISBLANK('CS SQL query'[CountryCode]) ),
    KEEPFILTERS( 'CS SQL query'[ExcludeIdFilter] = TRUE() ),
    KEEPFILTERS( NOT ISBLANK('CS SQL query'[FarmName]) ),
    KEEPFILTERS( NOT CONTAINSSTRING( LOWER('CS SQL query'[FarmName]), "netafim" ) ),
    KEEPFILTERS( NOT CONTAINSSTRING( LOWER('CS SQL query'[FarmName]), "test" ) ),
    KEEPFILTERS( NOT CONTAINSSTRING( LOWER('CS SQL query'[FarmName]), "demo" ) )
    )
"""


OUTPUT_CSV = "powerbi_daily_table.csv" 


def execute_dax_query():
    url = f"https://api.powerbi.com/v1.0/myorg/datasets/{DATASET_ID}/executeQueries"

    headers = {
        "Authorization": f"Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6InJ0c0ZULWItN0x1WTdEVlllU05LY0lKN1ZuYyIsImtpZCI6InJ0c0ZULWItN0x1WTdEVlllU05LY0lKN1ZuYyJ9.eyJhdWQiOiJodHRwczovL2FuYWx5c2lzLndpbmRvd3MubmV0L3Bvd2VyYmkvYXBpIiwiaXNzIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvMzUxMDI1MDItMzliYS00NGFjLWFjYzQtYzE4NWUwYmY5Mjg1LyIsImlhdCI6MTc2NDAwMTk3NywibmJmIjoxNzY0MDAxOTc3LCJleHAiOjE3NjQwMDY3MTEsImFjY3QiOjAsImFjciI6IjEiLCJhaW8iOiJBWlFBYS84YUFBQUF2Tk9hOTd6UFNvV0VEQWt0K3IrMEpaMFF6SVB1REJaV2pJR2t3ZUJwQStBUUxwVVlhMzFxS29YVktCSldqcGlNRzNRaE94d2xQOXFHcjNKZkVhdVYxTENxUjFzSElrOFZoR3Q4Skw2OGgzdFRJcUZENTNRYkRPaGZsa3FINTgzQXFJRHlUcUk3T21rMzB4Uml2VHJzaUswdGg0ai8zY1lBMzJvaHFIVTd6UnBPa3BQa21GSWJrT3BJSmhnZ2VlY2IiLCJhbXIiOlsicHdkIiwiZmlkbyIsInJzYSIsIm1mYSJdLCJhcHBpZCI6Ijg3MWMwMTBmLTVlNjEtNGZiMS04M2FjLTk4NjEwYTdlOTExMCIsImFwcGlkYWNyIjoiMCIsImRldmljZWlkIjoiZGJlZjYwNWEtZjNjYS00ODY4LWI2YzItNDZhNWZmZDNkZDNmIiwiZmFtaWx5X25hbWUiOiJBYnUgU2hrYXJhIiwiZ2l2ZW5fbmFtZSI6IlNlZ2FsIiwiaWR0eXAiOiJ1c2VyIiwiaXBhZGRyIjoiMTkyLjExNC4xMDUuMjQ2IiwibmFtZSI6IlNlZ2FsIGFidSBTaGthcmEgfCBPcmJpYSAoTmV0YWZpbSkiLCJvaWQiOiI0MjdlNTcwYi00NzFjLTQxMjgtOTM1NC1lYWI2MmQ0ZTM2NDkiLCJvbnByZW1fc2lkIjoiUy0xLTUtMjEtMTE3Mjc2NzAwMi01NzAxNzQ5ODMtMTg1MjkwMzcyOC0xMzAyMjQiLCJwdWlkIjoiMTAwMzIwMDQ1MTQwMEY1RSIsInJoIjoiMS5BUXNBQWlVUU5ibzVyRVNzeE1HRjRMLVNoUWtBQUFBQUFBQUF3QUFBQUFBQUFBQ0VBUHNMQUEuIiwic2NwIjoidXNlcl9pbXBlcnNvbmF0aW9uIiwic2lkIjoiMDA3YmEwODktZDQ0Yi03ZDA1LTg3YTUtYmU3NDczZTRiMGIxIiwic2lnbmluX3N0YXRlIjpbImR2Y19tbmdkIiwiZHZjX2RtamQiLCJrbXNpIl0sInN1YiI6Il9IUUJ5YTFjVFFnZUpwcGJ0cnNOeHh1TWFqRUhzWGk2UFVJQ05oMGVKSVkiLCJ0aWQiOiIzNTEwMjUwMi0zOWJhLTQ0YWMtYWNjNC1jMTg1ZTBiZjkyODUiLCJ1bmlxdWVfbmFtZSI6InNlZ2FsLmFidS5zaGthcmFAbmV0YWZpbS5jb20iLCJ1cG4iOiJzZWdhbC5hYnUuc2hrYXJhQG5ldGFmaW0uY29tIiwidXRpIjoiUTh1QVlKM2dNa0MtTjFscW1MSk1BQSIsInZlciI6IjEuMCIsIndpZHMiOlsiYjc5ZmJmNGQtM2VmOS00Njg5LTgxNDMtNzZiMTk0ZTg1NTA5Il0sInhtc19hY3RfZmN0IjoiNSAzIiwieG1zX2Z0ZCI6IkNPRXZhOFBNY0l1dmM5ZDFfdEYyS2xPQVFacFdqZUpocHFCU1JIc25qVmNCWlhWeWIzQmxibTl5ZEdndFpITnRjdyIsInhtc19pZHJlbCI6IjEgNCIsInhtc19zdWJfZmN0IjoiNCAzIn0.q6lLsUbtkLrI4vu-2mRNEP7WaxbI1YNZFRR5VQPQVBPiiQ1LJVkU9PyETRhHWIR7__SxAB9XJdSlJGtnAHXilqIVaqG1mex3JI3U4klPsc6_1oz1SZAJ1xJpNe2TX5QUgLIrp2Y-0xgVm7YXIfbd8T-5Un9_qePJt3iSTIZQe7hYLZ2_0V_FwcSRg5Cf7t-su5Jdpe-X7sy-bhYtn7SDwZ62RB0ia-q90zc8agngRD1QeQrahzAgVrErv8a77TRaYUsS9I9zGzty7zUyWmnqr7_271Nm35ZLf-gxy6-IdCx0IcACHixlJncE4x4Sthd38hWrmxWxEquLOvRg1QQwGA",
        "Content-Type": "application/json"
    }
    body = {
        "queries": [
            {"query": DAX_QUERY}
        ]
    }

    # Disable SSL verification for corporate proxy networks
    response = requests.post(url, headers=headers, json=body, verify=False)

    response.raise_for_status()
    return response.json()



def result_to_dataframe(result_json):
    """
    Convert the executeQueries JSON result to a pandas DataFrame.
    Assumes there is 1 result with 1 table.
    """
    tables = result_json["results"][0]["tables"]
    table = tables[0]

    columns = [c["name"] for c in table["columns"]]
    rows = table["rows"]

    df = pd.DataFrame(rows, columns=columns)
    return df

def update_daily_table():
    print("Calling Power BI executeQueries...")
    result = execute_dax_query()
    df = result_to_dataframe(result)

    print(f"Retrieved {len(df)} rows, columns: {list(df.columns)}")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved table to {OUTPUT_CSV}")


if __name__ == "__main__":
    update_daily_table()
