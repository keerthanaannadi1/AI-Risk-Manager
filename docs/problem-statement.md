# Problem Statement

## Overview

Merchants on digital payment platforms lose money silently. Unlike a robbery — which is visible and immediate — merchant losses from fraud, return abuse, and chargebacks happen slowly, through policy gaps, trust exploitation, and system blindspots.

This project attempts to surface these losses early using machine learning, and recommend defensive actions before the money is gone.

---

## Problem 1: Fraudulent Transactions

### What Happens

A bad actor obtains stolen card details (from a data breach, phishing, or dark web purchase). They use these details to place an order on a merchant's platform. The merchant ships the product. The real cardholder notices the unauthorized charge and disputes it with their bank. The bank reverses the payment. The merchant has already shipped the product and now loses both the goods and the revenue.

### Why It's Hard to Detect

- The transaction looks normal at the time of purchase
- Fraudsters use VPNs, residential proxies, and fresh device fingerprints to avoid detection
- AI tools now generate convincing synthetic identities, making fake accounts hard to distinguish from real ones
- Fraud patterns evolve constantly — models trained on old fraud get bypassed by new tactics

### Signals That Help

- Device or IP has never been seen on this platform before
- Shipping address is different from billing address
- Order value is significantly higher than the customer's usual spend
- Multiple orders placed in a short time window (velocity)
- Order placed minutes after account creation
- Payment method is a prepaid card (harder to trace)

### What I have Build

A **Fraud Transaction Scorer** that assigns a probability (0.0 to 1.0) to each transaction. Above a configured threshold, the transaction is flagged for review or step-up authentication.

---

## Problem 2: Return Abuse

### What Happens

Merchants offer return policies to reduce purchase hesitation. Some customers exploit this:

- **Wardrobing**: Buy a dress, wear it to an event, return it the next day
- **Swap fraud**: Buy a new phone, return the box with a broken/fake phone inside
- **Serial returners**: Return 70–90% of everything they ever buy — effectively using the merchant as a free rental service
- **Refund fishing**: Claim the item never arrived or was defective to get a refund without returning anything

### Why It's Hard to Detect

- Each individual return looks reasonable in isolation
- Merchants don't easily share return data across platforms (a serial returner on one site is unknown on another)
- Legitimate customers also return — high recall requirements mean you cannot just block everyone who returns frequently
- Return reasons are self-reported and easy to fake

### Signals That Help

- Customer's historical return rate (returns / total orders)
- Time between purchase and return request (very fast or very late = suspicious)
- Product category (electronics, luxury goods = higher swap risk)
- Whether images were submitted with the return
- Whether the customer contacted support before initiating the return
- Account age (new accounts returning immediately = suspicious)
- Number of returns in the last 30/60/90 days

### What I have Build

A **Return Risk Scorer** that assigns a risk level (Low / Medium / High) to each return request. High-risk returns are flagged for manual review or require photo proof before approval.

This is the **first module built** in this project because it has the most intuitive features, cleanest data structure, and clearest evaluation criteria.

---

## Problem 3: Chargebacks

### What Happens

A customer contacts their bank and disputes a charge — either because the card was actually stolen (true fraud) or because they want the product for free (friendly fraud). The bank initiates a chargeback. The payment gateway automatically reverses the payment. The merchant loses the money AND pays a chargeback fee (typically ₹500–₹2000 per case).

If a merchant's chargeback ratio exceeds ~1%, they risk losing their payment gateway account entirely — which means they can't accept online payments at all.

### Two Types

| Type | What It Is | How Common |
|------|-----------|------------|
| True fraud | Card was actually stolen | ~40% of chargebacks |
| Friendly fraud | Customer disputes a legitimate purchase | ~60% of chargebacks |

### Why It's Hard to Fight

- Banks default to the customer's side
- The evidence submission window is narrow (7–14 days)
- Merchants often don't know what evidence is needed or how to format it
- Fighting every chargeback manually is not scalable

### What I have Build

A **Chargeback Evidence Responder** that:
1. Automatically pulls relevant evidence (delivery confirmation, IP logs, device fingerprint, login history, communication records)
2. Scores the winnability of the dispute before investing time in it
3. Formats the evidence into a structured response document

