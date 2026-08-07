#!/usr/bin/env python3
"""
GA4 peak-load reporter.

Pulls minute-level traffic (page views, active users, events) from the GA4 Data
API and renders the busiest minutes as a table. Minute granularity (dateHourMinute)
is exposed by the API even though the GA4 web UI hides it.

Auth: relies on Application Default Credentials. Point GOOGLE_APPLICATION_CREDENTIALS
at your service-account JSON key, e.g.
    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/ga-credentials.json

Property: pass --property-id or set GA4_PROPERTY_ID in the environment.

Note: dateHourMinute values are in the property's configured timezone, not UTC.
"""

import calendar
import os
from datetime import UTC, date, datetime

import click
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    MetricAggregation,
    OrderBy,
    RunReportRequest,
)
from rich.console import Console
from rich.table import Table

# Map the friendly CLI sort keys to (api_field, is_dimension).
SORT_FIELDS = {
    "minute": ("dateHourMinute", True),
    "views": ("screenPageViews", False),
    "users": ("activeUsers", False),
    "events": ("eventCount", False),
    "sessions": ("sessions", False),
}

# (API metric name, column label). Order defines both the column order and the
# metric_values order in each row, so adding a metric here is the only edit needed.
METRICS = [
    ("screenPageViews", "Views"),
    ("activeUsers", "Active users"),
    ("eventCount", "Events"),
    ("sessions", "Sessions"),
]

# Length of a dateHourMinute value: YYYYMMDDHHMM.
MINUTE_STAMP_LENGTH = 12


def months_ago(n: int) -> str:
    """Return the ISO date n whole months before today, day-clamped for short months."""
    today = datetime.now(tz=UTC).date()
    total = today.year * 12 + (today.month - 1) - n
    year, month0 = divmod(total, 12)
    month = month0 + 1
    day = min(today.day, calendar.monthrange(year, month)[1])
    return date(year, month, day).isoformat()


def fmt_minute(raw: str) -> str:
    """202606141530 -> 2026-06-14 15:30 (falls back to raw if unexpected)."""
    if len(raw) == MINUTE_STAMP_LENGTH and raw.isdigit():
        return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]} {raw[8:10]}:{raw[10:12]}"
    return raw


def print_warnings(response, console) -> None:
    """Surface data-quality signals the API reports, on by default."""
    md = response.metadata
    for s in md.sampling_metadatas:
        space = int(s.sampling_space_size or 0)
        read = int(s.samples_read_count or 0)
        pct = (read / space * 100) if space else 0.0
        console.print(
            f"[yellow]\u26a0 Sampling applied[/yellow] — based on {read:,} of "
            f"{space:,} events ({pct:.1f}%); figures are estimates."
        )
    if md.subject_to_thresholding:
        console.print(
            "[yellow]\u26a0 Thresholding applied[/yellow] — some rows withheld for "
            "privacy (low user counts), so counts may be understated."
        )
    if md.data_loss_from_other_row:
        console.print(
            "[yellow]\u26a0 Cardinality data loss[/yellow] — some values were grouped "
            "into (other); minute-level detail may be incomplete."
        )


def build_order_by(sort_by: str, *, descending: bool) -> OrderBy:
    field, is_dimension = SORT_FIELDS[sort_by]
    if is_dimension:
        return OrderBy(
            dimension=OrderBy.DimensionOrderBy(
                dimension_name=field,
                order_type=OrderBy.DimensionOrderBy.OrderType.ALPHANUMERIC,
            ),
            desc=descending,
        )
    return OrderBy(
        metric=OrderBy.MetricOrderBy(metric_name=field),
        desc=descending,
    )


@click.command()
@click.option(
    "--property-id",
    default=lambda: os.environ.get("GA4_PROPERTY_ID", ""),
    help="Numeric GA4 property ID (or set GA4_PROPERTY_ID).",
)
@click.option(
    "--sort-by",
    type=click.Choice(list(SORT_FIELDS), case_sensitive=False),
    default="views",
    show_default=True,
    help="Column to sort by. Sorting is pushed to the API so --limit returns the true top-N.",
)
@click.option(
    "--order",
    type=click.Choice(["desc", "asc"], case_sensitive=False),
    default="desc",
    show_default=True,
    help="Sort direction.",
)
@click.option("--limit", default=20, show_default=True, help="Rows to return.")
@click.option(
    "--months",
    default=14,
    show_default=True,
    help="Look-back window in months. Defaults to the 14-month retention max; "
    "the range auto-covers whatever data exists within it.",
)
def main(property_id: str, sort_by: str, order: str, limit: int, months: int) -> None:
    """Report the busiest minutes for a GA4 property."""
    if not property_id:
        raise click.UsageError("Provide --property-id or set GA4_PROPERTY_ID.")

    descending = order.lower() == "desc"
    start_date = months_ago(months)

    client = BetaAnalyticsDataClient()  # reads GOOGLE_APPLICATION_CREDENTIALS
    request = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="dateHourMinute")],
        metrics=[Metric(name=name) for name, _ in METRICS],
        date_ranges=[DateRange(start_date=start_date, end_date="yesterday")],
        order_bys=[build_order_by(sort_by, descending=descending)],
        metric_aggregations=[MetricAggregation.MAXIMUM],
        limit=limit,
    )
    response = client.run_report(request)

    console = Console()
    table = Table(
        title=f"GA4 busiest minutes  ·  {start_date} → yesterday  ·  sorted by {sort_by} {order.lower()}",
        caption=f"{len(response.rows)} of {response.row_count} matching minutes shown",
    )
    table.add_column("Timestamp", style="cyan", no_wrap=True)
    for _, label in METRICS:
        table.add_column(label, justify="right")

    for row in response.rows:
        ts = fmt_minute(row.dimension_values[0].value)
        table.add_row(ts, *(mv.value for mv in row.metric_values))

    console.print(table)

    # metric_aggregations=MAXIMUM computes the peak of each metric across the full
    # result set (not just the returned rows), so it stays accurate under --limit.
    if response.maximums:
        peak = response.maximums[0].metric_values
        parts = ", ".join(f"{label.lower()}: {mv.value}" for (_, label), mv in zip(METRICS, peak, strict=False))
        console.print(f"[bold]Peak values across range[/bold] — {parts}")

    print_warnings(response, console)

    if response.property_quota and response.property_quota.tokens_per_day.consumed:
        q = response.property_quota.tokens_per_day
        console.print(f"[dim]API tokens today: {q.consumed}/{q.consumed + q.remaining}[/dim]")


if __name__ == "__main__":
    main()
