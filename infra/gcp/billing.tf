# Sign-off gate for real recurring GCP cost -- a notification threshold,
# not a pre-spend block like cost_ledger.py's Claude-spend ceiling (GCP
# billing isn't triggered by a call this codebase controls). Empty
# var.billing_account_id skips this -- creating a budget needs
# billing-account-level IAM the project-scoped terraform identity may not
# hold; see infra/DEPLOYMENT.md for how to apply it separately if so.
resource "google_billing_budget" "phase_d_monthly" {
  count           = var.billing_account_id == "" ? 0 : 1
  billing_account = var.billing_account_id
  display_name    = "CausalOps hosted-deployment monthly ceiling"

  budget_filter {
    projects = ["projects/${data.google_project.current.number}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.budget_monthly_usd)
    }
  }

  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 1.0
  }
}
