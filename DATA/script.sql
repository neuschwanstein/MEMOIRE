CREATE DATABASE feed;

CREATE TABLE feed
(
id SERIAL,
ticker VARCHAR(12),
href VARCHAR(1000),
published DATE,
title VARCHAR(1000),
engagement INT,
engagement_rate float8,
provider VARCHAR(1000),
content TEXT
);
