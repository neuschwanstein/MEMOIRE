(* ::Package:: *)

(* Date methods *)


(* Yahoo xml access *)

(* Returns 
	- CompanyName (?)
	- start
	- end
	- Sector
	- Industry
	- FullTimeEmployees *)
GetYahooStockInfoAssociation[ticker_]:=
Module[{url,xml,result},
	url = URLBuild["https://query.yahooapis.com/v1/public/yql",
			{"q"->"select * from yahoo.finance.stocks where symbol=\""<>ticker<>"\"",
			 "diagnostics"->"true",
			 "env"->"store://datatables.org/alltableswithkeys"}
			];

	xml=Import[url,"XML"];

	MapAt[DateList,
		Association@Flatten@Cases[
			Cases[xml,XMLElement["stock",_,val_]->val,Infinity],
			XMLElement[key_,_,{val_}]->{key->val},
			Infinity],
		List /@ Key /@ {"start","end"}]
];

GetYahooHistoricalInfo[ticker_, startDate_]:=
Module[{createStartEndList, yqlQuery, getHistoricalInfoInternal},

	createStartEndList[date_List]:=
	Module[{year,followingJan1st,getStartingDates},

		year[d_List] := d[[1]];
		followingJan1st[d_List] := DateList[{year[d]+1, 1, 1}];

		getStartingDates[d_List] :=
		With[{endDate = DateList[Today]},
			NestWhileList[followingJan1st, d, Function[\[Delta], year[\[Delta]]!=year[endDate]]] ~ Join ~ {endDate}
		];

		With[{startingDates = getStartingDates[date]},
			{Most[startingDates],Most@Rest@startingDates~Join~{DateList[Today]}}
		]
	];

	yqlQuery[symbol_,internalStartDate_,endDate_]:=
	With[{dateFormat = {"Year","-","Month","-","Day"}},
		"SELECT * FROM yahoo.finance.historicaldata WHERE symbol=\""<>symbol<>"\" AND startDate=\""<>DateString[internalStartDate,dateFormat]<>"\" AND endDate=\""<>DateString[endDate,dateFormat]<>"\""
	];

	getHistoricalInfoInternal[internalStartDate_,internalEndDate_]:=
	Module[{url,xml},
		url=URLBuild["https://query.yahooapis.com/v1/public/yql",
			{"q"->yqlQuery[ticker,internalStartDate,internalEndDate],
			 "diagnostics"->"true",
			 "env"->"store://datatables.org/alltableswithkeys"}
		];

		xml=Import[url,"XML"];

		Association/@Map[
			Cases[#,XMLElement[key_,_,{val_}]->key->val]&,
			Cases[xml,XMLElement["quote",_,val_]->val,Infinity]
		]
	];

	Flatten[MapThread[getHistoricalInfoInternal,createStartEndList[startDate]]]
];


(* DB Connection methods *)
Needs["DatabaseLink`"];

$DBCONNECTIONSTRING = {"PostgreSQL", "localhost/ProjetC"};

(*
			ADD OVERLOADING HERE ...
AddHistoricDatasetToDatabase[ticker_String, {__dataset}]:=
	AddHistoricDatasetToDatabase
*)
AddHistoricDatasetToDatabase[ticker_String, {dataset__Association}] :=
With[{conn=OpenSQLConnection[JDBC @@ $DBCONNECTIONSTRING],
		keys=Normal@Keys@First@dataset,
		tableName="HISTORIC"},

	SQLInsert[conn, tableName, {"Ticker"}~Join~keys,
		ArrayFlatten[{{ticker, Values @ dataset[All,{"Date"->SQLDateTime}]}}]
	];
	CloseSQLConnection[conn];
];

AddHistoricDatasetToDatabase[ticker_String,dataset_Dataset]:=
	AddHistoricDatasetToDatabase[ticker, Normal@dataset];
