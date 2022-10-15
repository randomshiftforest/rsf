use crate::{
    prelude::*,
    tests::{graphs::ROOT, utils::save_jpeg},
};
use chrono::{NaiveDate, NaiveDateTime};
use itertools::Itertools;
use ndarray::prelude::*;
use plotly::{
    color::NamedColor,
    common::Title,
    layout::{Axis, Shape, ShapeLayer, ShapeLine, ShapeType},
    Layout, Plot, Scatter,
};
use serde::{Deserialize, Deserializer};

#[derive(Debug, Deserialize)]
struct TwitterRec {
    #[serde(deserialize_with = "de_twitter_date")]
    timestamp: i64,
    token1: String,
    token2: String,
}

fn de_twitter_date<'de, D>(deserializer: D) -> Result<i64, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    NaiveDateTime::parse_from_str(&s, "%m:%d:%Y:%H:%M:%S")
        .map(|t| t.timestamp())
        .map_err(serde::de::Error::custom)
}

#[derive(Deserialize)]
struct TwitterWorldCupLabel {
    #[serde(rename = "venue")]
    _venue: String,
    #[serde(rename = "teams")]
    _teams: String,
    #[serde(rename = "players_involved")]
    _players_involved: String,
    #[serde(rename = "event")]
    _event: String,
    date: String,
    #[serde(rename = "importance")]
    _importance: String,
}

#[test]
fn twitter_world_cup() {
    let dt = 3600;

    let y_rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b';')
        .from_path("in/graph/twitter/twitter_wc_labels.csv")
        .unwrap();
    let labels = y_rdr
        .into_deserialize::<TwitterWorldCupLabel>()
        .filter_map(Result::ok)
        // .filter(|rec| rec.importance.starts_with("High"))
        .filter_map(|rec| {
            NaiveDateTime::parse_from_str(&rec.date, "%_m/%_d/%Y %H:%M:%S")
                .map(|dt| dt.timestamp())
                .ok()
        })
        .map(|timestamp| timestamp / dt);

    let x_rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b' ')
        .from_path("in/graph/twitter/twitter_world_cup.txt")
        .unwrap();
    let groups = x_rdr
        .into_deserialize::<TwitterRec>()
        .filter_map(Result::ok)
        .group_by(|rec| rec.timestamp / dt);
    let (keys, graphs): (Vec<_>, Vec<_>) = groups
        .into_iter()
        .map(|(key, group)| {
            let edges: Vec<(String, String, f32)> =
                group.map(|rec| (rec.token1, rec.token2, 1.0)).collect();
            let graph = Graph::new(edges);
            (key, graph)
        })
        .unzip();
    println!(
        "#edges: {}",
        graphs.iter().map(|g| g.edge_iter().count()).sum::<usize>()
    );
    println!("#graphs {}", graphs.len());

    let sl_cfg = SpotLightConfig::new(50, 0.2, 0.2);
    let sketches: Vec<_> = graphs.into_iter().spotlight(&sl_cfg).collect();

    let bb = sketches.iter().map(|sketch| sketch.view()).bb().unwrap();
    let rsf_cfg = ConfigBuilder::default()
        .bounding_box(bb)
        .n_points(64)
        .n_trees(64)
        .build();
    let n = sketches.len();
    let y_pred: Array1<_> = sketches
        .into_iter()
        .rsf_window::<true>(&rsf_cfg)
        .transform(&rsf_cfg)
        .collect();
    let shift = n - y_pred.len();

    let (first_key, last_key) = (keys.first().unwrap(), keys.last().unwrap());
    let shifted_keys: Vec<_> = keys.iter().map(|key| key - first_key).collect();
    let shifted_labels: Vec<_> = labels.into_iter().map(|label| label - first_key).collect();
    let mut plot = Plot::new();
    plot.add_trace(Scatter::new(shifted_keys[shift..n].to_vec(), y_pred));
    let mut layout = Layout::new()
        .x_axis(
            Axis::new()
                .title(Title::new("time (hours)"))
                .range(vec![0, last_key - first_key]),
        )
        .y_axis(Axis::new().title(Title::new("anomaly score")));
    for label in shifted_labels {
        layout.add_shape(
            Shape::new()
                .x_ref("x")
                .y_ref("paper")
                .shape_type(ShapeType::Rect)
                .x0(label)
                .y0(0f64)
                .x1(label + 1)
                .y1(1f64)
                .fill_color(NamedColor::LightSalmon)
                .opacity(0.5)
                .layer(ShapeLayer::Below)
                .line(ShapeLine::new().width(0.)),
        );
    }
    plot.set_layout(layout);
    save_jpeg(
        &format!("{ROOT}/twitter"),
        "twitter_world_cup_2014",
        plot,
        900,
        450,
    );
}

#[derive(Debug, Deserialize)]
struct TwitterSecurityLabel {
    date: String,
    #[serde(rename = "event_type")]
    _event_type: String,
    #[serde(rename = "entities")]
    _entities: String,
    #[serde(rename = "event_description")]
    _event_description: String,
    #[serde(rename = "importance")]
    _importance: String,
}

#[test]
fn twitter_security() {
    let dt = 3600;

    let y_rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b';')
        .from_path("in/graph/twitter/twitter_security_labels.csv")
        .unwrap();
    let labels = y_rdr
        .into_deserialize::<TwitterSecurityLabel>()
        .map(|rec| rec.unwrap())
        .map(|rec| {
            NaiveDate::parse_from_str(&rec.date, "%_m/%_d/%y")
                .map(|dt| dt.and_hms(12, 0, 0).timestamp())
                .unwrap()
        })
        .map(|timestamp| timestamp / dt);

    let x_rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b' ')
        .from_path("in/graph/twitter/twitter_security.txt")
        .unwrap();
    let groups = x_rdr
        .into_deserialize::<TwitterRec>()
        .filter_map(Result::ok)
        .group_by(|rec| rec.timestamp / dt);
    let (keys, graphs): (Vec<_>, Vec<_>) = groups
        .into_iter()
        .map(|(key, group)| {
            let edges: Vec<(String, String, f32)> =
                group.map(|rec| (rec.token1, rec.token2, 1.0)).collect();
            let graph = Graph::new(edges);
            (key, graph)
        })
        .unzip();
    println!(
        "#edges: {}",
        graphs.iter().map(|g| g.edge_iter().count()).sum::<usize>()
    );
    println!("#graphs {}", graphs.len());

    let sl_cfg = SpotLightConfig::new(50, 0.2, 0.2);
    let sketches: Vec<_> = graphs.into_iter().spotlight(&sl_cfg).collect();

    let bb = sketches.iter().map(|sketch| sketch.view()).bb().unwrap();
    let rsf_cfg = ConfigBuilder::default()
        .bounding_box(bb)
        .n_points(64)
        .n_trees(64)
        .build();
    let n = sketches.len();
    let y_pred: Array1<_> = sketches
        .into_iter()
        .rsf_window::<true>(&rsf_cfg)
        .transform(&rsf_cfg)
        .collect();
    let shift = n - y_pred.len();

    let (first_key, last_key) = (keys.first().unwrap(), keys.last().unwrap());
    let shifted_keys: Vec<_> = keys.iter().map(|key| key - first_key).collect();
    let shifted_labels: Vec<_> = labels.into_iter().map(|label| label - first_key).collect();
    let mut plot = Plot::new();
    plot.add_trace(Scatter::new(shifted_keys[shift..n].to_vec(), y_pred));
    let mut layout = Layout::new()
        .x_axis(
            Axis::new()
                .title(Title::new("time (hours)"))
                .range(vec![0, last_key - first_key]),
        )
        .y_axis(Axis::new().title(Title::new("anomaly score")));
    for label in shifted_labels {
        layout.add_shape(
            Shape::new()
                .x_ref("x")
                .y_ref("paper")
                .shape_type(ShapeType::Rect)
                .x0(label - 12)
                .y0(0f64)
                .x1(label + 12)
                .y1(1f64)
                .fill_color(NamedColor::LightSalmon)
                .opacity(0.5)
                .layer(ShapeLayer::Below)
                .line(ShapeLine::new().width(0.)),
        );
    }
    plot.set_layout(layout);
    save_jpeg(
        &format!("{ROOT}/twitter"),
        "twitter_security_2014",
        plot,
        900,
        450,
    );
}
