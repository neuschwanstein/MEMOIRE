CREATE DATABASE feed;

CREATE TABLE reuters_globalMarketNews
(
id SERIAL,
feedly_id text UNIQUE,
href text,
published DATE,
title TEXT,
engagement INT,
engagement_rate float8,
content TEXT
);
