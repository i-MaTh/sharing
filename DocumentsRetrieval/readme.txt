**All the files to be placed at the same folder.

**Run the scripts in terminal on your platforms

**Usage of doc_retrieval.py
	USE: python <PROGNAME>  -d DOCUMENTS -q QUERIES -s STOP_LIST -i INDEX -o OUTPUT
        -h : print this help message
        -d <file> : input <file> as documents
        -q <file> : input <file> as queries
        -s <file> : input <file> as stop_list
        -i <file> : if inverted index <file> does not exist, building a new one to <file>.
        -o <file> : output result to <file>
        -k <int> : top-k relevant results for each query
        
    Examples: python doc_retrieval.py -d documents.txt -q queries.txt -s stop_list.txt -i inverted_index.txt -o output.txt -k 10
