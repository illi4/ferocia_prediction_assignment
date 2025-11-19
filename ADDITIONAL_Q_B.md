# Part B additional questions

## 1. If there is a model already in production, what are your thought process behind how you would promote the model safely and ensure that it is still performant?

Overall, I would suggest using simpler approaches for internal tools with known consumers, and stringent multi-stage rollouts for high-traffic public APIs.  

**Offline validation gate**  as the first step. I already discussed this in the part A, but still, the new model must demonstrate: 
- Performance meeting minimum thresholds and not degrading relative to the current production model on a fixed holdout set
- Consistent performance across customer segments (age groups, job types, balance brackets)  
- Expected prediction distributions  

**Shadow deployment**: 
- Deploy the new model alongside production so both score identical requests, but only the production model's predictions are returned to consumers. 
- Run for 1-2 weeks collecting data across different patterns.
- Compare prediction distributions, latency under load, and actual performance metrics based on ground truth if possible. 

**Gradual rollout with the possibility of rollback** 
- Route increasing percentages of traffic to the new model over days or weeks, monitoring closely at each step. 
- Critical: maintain ability to instantly revert to the previous model if any metric degrades (e.g. using MLflow). 
- The API should detect registry changes and be able to reload models. This is not supported in the current implementation as the model only loads on startup.

**Additionally to consider**: A/B testing with statistical significance testing, comprehensive monitoring dashboards, approval gates requiring human review. 

## 2. You're ready to deploy a new model, but the schema and preprocessing code is different to the current model in production - how would you handle this change? What would you do about the existing consumers of the API?

**For internal APIs with known consumers** (probably our case), the simplest approach is coordinated upgrade: Directly contact all consumers (marketing team, etc.), agree on a maintenance window, deploy the new model and schema simultaneously, and provide clear migration documentation. This works well when you have 2-5 known consumers you can coordinate with. The risk is lower because you control both sides of the integration and can verify migrations before deployment.

**For external APIs or unknown consumers**, API versioning is necessary. I would suggest to introduce `/v2/predict` alongside existing `/v1/predict`, each with separate Pydantic models, preprocessing pipelines, and model artefacts.
With the current FastAPI structure, this could be `models_v2.py` and `predictor_v2.py` and a new endpoint. I would, however, rework the folder structure and explore best practices around code organisation for this scenario as naming a file `_v2` does not seem like a great engineering idea. 
Consumers then can migrate at their own pace or through coordinated effort.  

For either approach, clear communication and testing considerations are critical: 
- Detailed changelog (what changed and why)
- Side-by-side schema comparison
- Code examples showing the migration
- Explicit testing period in staging
- Testing endpoints where consumers validate their v2 payloads before switching
- Deprecation timeline for v1 (with response headers like `X-API-Deprecation: v1 deprecated on 2026-XX-XX`)
- Track v1 usage via CloudWatch or other tool

## 3. In an ideal world, what sort of observability metrics would you have on the API? When would you trigger an alert?

I would plan out to build these in phases starting with essentials and adding sophistication as the system matures. For this prediction API, ground truth probably arrives weeks or months after predictions (customers eventually subscribe or don't), which would also shape the monitoring strategy. 

### Phase 1: Operational health (most important)

**Latency metrics**: 
- p50, p95, and p99 latency for the `/predict` endpoint with separate breakdowns for preprocessing time vs model inference time vs serialisation
- Alert if p95 > a specific lower threshold (e.g. 500ms) (warning - user experience degrading) or p99 > 1000ms (critical - severe issues)

**Error rates and throughput**: 
- HTTP status codes (200, 4xx, 5xx)
- Error rate as 5xx errors divided by total requests
- Requests per second. 
- Alert critically if error rate > 5% sustained for 5 minutes (API effectively down), warn if error rate > 1% for 15 minutes (degradation starting), and investigate if throughput suddenly drops to zero (upstream consumer failure)
- Existing `/health` endpoint should be monitored every 30 seconds - alert critically if it fails 3 consecutive checks.

**Resource utilisation** : 
- CPU and memory usage per container, with alerts at CPU > 80% sustained for 10 minutes (scale up needed) and memory > 90% (potential OOM kills)
- Specific validation error rate (422 responses from Pydantic) - if this suddenly exceeds 5%, upstream data sources probably have changed format.

Implementation: FastAPI middleware logs every request with timestamp, latency, status code, prediction, and probability as structured JSON to CloudWatch Logs. CloudWatch Metrics aggregates these into dashboards. CloudWatch Alarms trigger SNS → PagerDuty for critical alerts, Slack for warnings.

### Phase 2: Model behaviour monitoring  

Since ground truth is delayed, **prediction distribution monitoring** would likely become the primary model health indicator: 
- Percentage of "yes" vs "no" predictions over time (baseline ~11% from training data)
- Average prediction probability (model confidence)
- Distribution of confidence levels from API responses (low/medium/high). 
- Alert if "yes" prediction rate shifts > 20% absolute from baseline (something fundamental changed?) or if average probability drifts toward 0.5 (model becoming uncertain about everything).

**Input data distribution monitoring** to detect drift before it degrades performance: 
- Population Stability Index (PSI) daily comparing incoming feature distributions to training data distributions. 
- Alert if PSI > 0.2 for any feature (significant drift requiring investigation) or PSI > 0.1 for multiple features (broad distributional change). 
- Track specific statistics: mean age, balance distribution, job category frequencies.  

**Validation patterns** : 
- Pydantic models with strict `Literal` types reject unknown categorical values - these could be monitored too, this signals schema drift. We can log these gracefully with warnings rather than hard failures where possible, giving visibility without breaking the API.

Implementation: Lambda function that runs hourly, pulls CloudWatch Logs, computes drift metrics, publishes back to CloudWatch Metrics. MLflow to log daily aggregates for historical trending.

### Phase 3: Delayed performance validation (weekly/monthly)

**Ground truth evaluation** in batches when labels arrive: 
- Weekly or monthly, join predictions with eventual outcomes (did customer subscribe?) and compute F1, precision, recall etc. 
- Alert if F1 drops > 10% relative to the baseline model (performance degradation)

**Business metrics**: 
- Conversion rates (of "yes" predictions, what percentage subscribed?)
- False negative rate (of "no" predictions, what percentage unexpectedly subscribed?)
- Campaign ROI
- Alert if conversion rates drop > 15% relative to historical baselines

### Alert prioritisation

**Critical alerts** (alert immediately): error rate > 5%, p99 latency > 2000ms, health check failing, model failed to load, zero throughput for 10 minutes.  

**Warning alerts** (Slack during business hours): error rate > 1%, p95 latency > 500ms, CPU > 80% sustained, prediction distribution shifts > 15%, PSI > 0.2 for any feature. Indicate degradation starting or drift requiring investigation.

**Information alerts** (daily email summary or daily Slack message): slight latency increases, traffic pattern changes, minor drift (PSI > 0.1), new categorical values appearing. For review but not needed to act immediately. 

**Synthetic monitoring** to run every X minutes: Lambda calls `/predict` with test data, verifies latency and sensible predictions. This can detect issues before real users do.

*Also to consider: calibration monitoring (do predicted probabilities match actual subscription rates?), segment-specific performance breakdowns (does the model perform worse for certain job types or age groups?), request pattern anomalies for security, and auto-scaling triggers based on resource metrics.*
