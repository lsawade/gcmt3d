# Download format and FetchData

The data downloaded has to be in a certain format to make a successful request. The `gcmt3d.data_download.dataRequest`
class will create a `request.txt` in the directory of the earthquake that has following columns:

`Network Station Location Channel StartTime EndTime`

So, a request file can look somewhat like this:

```
II AAK 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
II ABKT 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
II ABPO 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
II ALE 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
II ARU 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
II ASCN 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
II BFO 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
II BORG 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
IU ADK 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
IU AFI 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
IU ANMO 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
IU ANTO 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
IU BBSR 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
IU BILL 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
IU CASY 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
IU CCM 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
IU CHTO 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
IU COLA 00 BHZ 2011-01-01T00:00:00 2011-01-01T01:00:00
```

## Download commands

There are several download commands that I should know and add

### Download miniseed including response with selection file

```bash
#
```

### Download SAC including response with selection file

```bash
FetchData -l request.txt -o mydata.mseed -sd <directory>
```


